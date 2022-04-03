import os
import json
from typing import Optional, Tuple, Dict, NoReturn
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import transformers
from transformers import BartTokenizerFast
from transformers.models.bart.configuration_bart import BartConfig

from arguments import add_train_args, add_predict_args, add_wandb_args
import models
from inference import predict, get_top_k_sentences, extract_sentences, generate_summary
from utils import set_all_seeds, collate_fn, freeze, unfreeze_all, np_sigmoid, compute_rouge_l
from dataset import SummaryDataset
from torch.utils.data import DataLoader

from tqdm.auto import tqdm

def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

def train_step(args, model, tokenizer, batch, device) -> Tuple[torch.FloatTensor, Dict[str, float]]:

    if batch["labels"] is not None:
        ext_input_ids = shift_tokens_right(
            batch["input_ids"], model.config.pad_token_id, model.config.decoder_start_token_id,
        )
        gen_input_ids = shift_tokens_right(
            batch["labels"], model.config.pad_token_id, model.config.decoder_start_token_id,
        )

    input_ids = batch["input_ids"].to(device)  # (B, L_src)
    attention_mask = batch["attention_mask"].to(device)  # (B, L_src)
    answers = batch["answers"].to(device) # 추출요약 (B, 3)
    labels = batch["labels"].to(device) # 생성요약 (B, L_tgt)

    B = input_ids.size(0)
    MAX_NUM = torch.max(input_ids.eq(model.config.eos_token_id).sum(1))

    encoder_out = model.model.encoder(input_ids=input_ids, attention_mask=attention_mask)
    ext_decoder_out = model.model.decoder(
        input_ids=ext_input_ids.to(device), 
        encoder_hidden_states=encoder_out[0], 
        encoder_attention_mask=attention_mask,
    )
    gen_decoder_out = model.model.decoder(
        input_ids=gen_input_ids.to(device),
        encoder_hidden_states=encoder_out[0],
        encoder_attention_mask=attention_mask,
    )

    ext_hidden_states = ext_decoder_out[0]  # [B, L, D]
    gen_hidden_states = gen_decoder_out[0]
    
    # extraction part
    ext_logits = model.classification_head(ext_hidden_states).squeeze(-1) # [B, L]
    logits = torch.full((B, MAX_NUM), -1e9, dtype=torch.float).to(device) # [B, MAX_NUM]
    for i in range(B):
        _logit = ext_logits[i][input_ids[i].eq(model.config.eos_token_id)]
        l = _logit.size(0)
        logits[i, 0:l] = _logit
    
    one_hot = torch.zeros((B, MAX_NUM)).to(device)
    for i in range(B):
        one_hot[i,:].index_fill_(0, answers[i][answers[i] >= 0], 1.0)
    ext_labels = one_hot.clone()

    ext_loss_fn = nn.BCEWithLogitsLoss()
    ext_loss = ext_loss_fn(logits, ext_labels) # [B]

    # generation part
    gen_logits = model.lm_head(gen_hidden_states) + model.final_logits_bias
    gen_loss_fn = nn.CrossEntropyLoss(reduction='none')
    gen_loss = gen_loss_fn(gen_logits.view(-1, model.config.vocab_size), labels.view(-1)) # [B*L]
    gen_loss = gen_loss.view(B, -1) # [B, L]
    gen_loss = gen_loss.mean(dim=1) # [B]

    
    if args.freeze_backbone:
        total_loss = ext_loss
        metrics = {"ext_loss": ext_loss.item(), "ext_logits": logits}
    else:
        total_loss = args.loss_alpha * ext_loss + (1-args.loss_alpha) * gen_loss.mean()
        metrics = {"ext_loss": ext_loss.item(), "gen_loss": gen_loss.mean().item(), "ext_logits": logits}

    # if using prediction module
    if args.prediction_module is not None:
        pred_out = model.predict_module(ext_hidden_states) # [B]
        if args.prediction_module.lower() == "lpm":
            target_out = gen_loss.clone().detach() # target of loss prediction module; [B]
        elif args.prediction_module.lower() == "rpm":
            model.eval()
            with torch.no_grad():
                top_ext_ids = get_top_k_sentences(
                    logits=logits.clone().detach().cpu(), 
                    eos_positions=batch["eos_positions"], 
                    k = args.top_k,
                )
                batch = extract_sentences(batch["input_ids"], batch["eos_positions"], top_ext_ids, tokenizer)
                generated_ids = generate_summary(args, model, batch, device)
            REMOVE_IDS = np.array([tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id, model.config.decoder_start_token_id])
            target_out = compute_rouge_l(generated_ids.cpu().numpy(), labels.cpu().numpy(), REMOVE_IDS)["f1"]  # set Rouge-L F1 score as target output
            target_out = torch.from_numpy(target_out).to(device)  # target of rouge prediction module; [B]
        
        if args.pred_loss_function is not None and args.pred_loss_function.lower() == "l1":
            pred_loss_fn = nn.L1Loss()
        else:
            pred_loss_fn = nn.MSELoss()

        pred_loss = pred_loss_fn(pred_out, target_out)
        total_loss += args.loss_beta * pred_loss
        metrics.update({"pred_loss": pred_loss.item()})

    return total_loss, metrics

def train_loop(args, model, tokenizer, train_dl, eval_dl, optimizer, prev_step: int = 0) -> int:
    
    step = prev_step

    model.train()
    optimizer.zero_grad()
    ext_losses = []
    gen_losses = []
    pred_losses = []
    all_logits = []

    if args.use_wandb:
        import wandb

    if args.do_train:
        tqdm_bar = tqdm(train_dl)
        for batch in tqdm_bar:

            model.train()
            device = torch.device("cpu") if args.no_cuda or not torch.cuda.is_available() else torch.device("cuda")

            loss, returned_dict = train_step(args, model, tokenizer, batch, device)
            loss.backward()
            ext_losses.append(returned_dict["ext_loss"])
            if not args.freeze_backbone:
                gen_losses.append(returned_dict["gen_loss"])
            if args.prediction_module is not None:
                pred_losses.append(returned_dict["pred_loss"])
            all_logits.append(returned_dict["ext_logits"].detach().cpu().numpy().flatten())
            step += 1

            if (step+1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                all_logits = np.hstack(all_logits)
                all_probs = np_sigmoid(all_logits)
                hist = np.histogram(all_probs)

                train_metrics = {
                    "train/ext_loss": np.mean(ext_losses), 
                    "train/gen_loss": np.mean(gen_losses), 
                    "train/pred_loss": np.mean(pred_losses), 
                    "train/probs": wandb.Histogram(np_histogram=hist),
                    "step": step,
                }
                if args.use_wandb:
                    wandb.log(train_metrics)

                ext_losses = []
                gen_losses = []
                pred_losses = []
                all_logits = []

            if args.do_eval and (step+1) % args.eval_steps == 0:
                eval(args, model, tokenizer, eval_dl, step)

            tqdm_bar.set_description(f"Train step {step} ext_loss {np.mean(ext_losses):.3f} gen_loss {np.mean(gen_losses):.3f} pred_loss {np.mean(pred_losses):.3f}")

    return step


def eval(args, model, tokenizer, eval_dl, step) -> Dict[str, float]:
    device = torch.device("cpu") if args.no_cuda or not torch.cuda.is_available() else torch.device("cuda")
    eval_metrics = eval_loop(model, tokenizer, eval_dl, device)
    eval_metrics = {("eval/" + k): v for k, v in eval_metrics.items()}
    eval_metrics["step"] = step

    print(eval_metrics)
    if args.use_wandb:
        import wandb
        wandb.log(eval_metrics)
    
    return eval_metrics


def eval_loop(model, tokenizer, eval_dl, device) -> Dict[str, float]:

    if args.use_wandb:
        import wandb

    model.eval()

    ext_loss = 0.0
    gen_loss = 0.0
    pred_loss = 0.0
    all_logits = []
    n = 0

    with torch.no_grad():
        for batch in tqdm(eval_dl):
            if batch["labels"] is not None:
                ext_input_ids = shift_tokens_right(
                    batch["input_ids"], model.config.pad_token_id, model.config.decoder_start_token_id,
                )
                gen_input_ids = shift_tokens_right(
                    batch["labels"], model.config.pad_token_id, model.config.decoder_start_token_id,
                )

            input_ids = batch["input_ids"].to(device)  # (B, L_src)
            attention_mask = batch["attention_mask"].to(device)  # (B, L_src)
            answers = batch["answers"].to(device) if "answers" in batch.keys() else None # 추출요약 (B, 3)
            labels = batch["labels"].to(device) if "labels" in batch.keys() else None

            B = input_ids.size(0)
            MAX_NUM = torch.max(input_ids.eq(model.config.eos_token_id).sum(1))

            encoder_out = model.model.encoder(input_ids=input_ids, attention_mask=attention_mask)
            ext_decoder_out = model.model.decoder(
                input_ids=ext_input_ids.to(device), 
                encoder_hidden_states=encoder_out[0], 
                encoder_attention_mask=attention_mask,
            )
            gen_decoder_out = model.model.decoder(
                input_ids=gen_input_ids.to(device),
                encoder_hidden_states=encoder_out[0],
                encoder_attention_mask=attention_mask,
            )

            ext_hidden_states = ext_decoder_out[0]  # last hidden state [B, L, D]
            gen_hidden_states = gen_decoder_out[0]
            
            # extraction part
            ext_logits = model.classification_head(ext_hidden_states).squeeze(-1) # [B, L]
            logits = torch.full((B, MAX_NUM), -1e9, dtype=torch.float).to(device) # [B, MAX_NUM]
            for i in range(B):
                _logit = ext_logits[i][input_ids[i].eq(model.config.eos_token_id)]
                l = _logit.size(0)
                logits[i, 0:l] = _logit
            
            one_hot = torch.zeros((B, MAX_NUM)).to(device)
            for i in range(B):
                one_hot[i,:].index_fill_(0, answers[i][answers[i] >= 0], 1.0)
            ext_labels = one_hot.clone()

            ext_loss_fn = nn.BCEWithLogitsLoss()
            ext_loss_b = ext_loss_fn(logits, ext_labels) # [B]

            # generation part
            gen_logits = model.lm_head(gen_hidden_states) + model.final_logits_bias
            gen_loss_fn = nn.CrossEntropyLoss(reduction='none')
            gen_loss_b = gen_loss_fn(gen_logits.view(-1, model.config.vocab_size), labels.view(-1)) # [B*L]
            gen_loss_b = gen_loss_b.view(B, -1) # [B, L]
            gen_loss_b = gen_loss_b.mean(dim=1) # [B]

            all_logits.append(logits.cpu().numpy().flatten())

            # weighted sum
            size = len(input_ids)
            ext_loss = (n * ext_loss + size * ext_loss_b.item()) / (n + size)
            gen_loss = (n * gen_loss + size * gen_loss_b.mean().item()) / (n + size)

            # if using prediction module
            if args.prediction_module is not None:
                pred_out = model.predict_module(ext_hidden_states) # [B]
                if args.prediction_module.lower() == "lpm":
                    target_out = gen_loss_b.clone().detach() # target of loss prediction module; [B]
                elif args.prediction_module.lower() == "rpm":
                    top_ext_ids = get_top_k_sentences(
                        logits=logits.clone().detach().cpu(), 
                        eos_positions=batch["eos_positions"], 
                        k = args.top_k,
                    )
                    batch = extract_sentences(batch["input_ids"], batch["eos_positions"], top_ext_ids, tokenizer)
                    generated_ids = generate_summary(args, model, batch, device)
                    REMOVE_IDS = np.array([tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id, model.config.decoder_start_token_id])
                    target_out = compute_rouge_l(generated_ids.cpu().numpy(), labels.cpu().numpy(), REMOVE_IDS)["f1"]  # set Rouge-L F1 score as target output
                    target_out = torch.from_numpy(target_out).to(device)  # target of rouge prediction module; [B]
                
                pred_loss_fn = nn.MSELoss()
                pred_loss_b = pred_loss_fn(pred_out, target_out)
                # weighted sum
                pred_loss = (n * pred_loss + size * pred_loss_b.item()) / (n + size)

            n += size

    all_logits = np.hstack(all_logits)
    all_probs = np_sigmoid(all_logits)
    hist = np.histogram(all_probs)

    return {
        "ext_loss": ext_loss,
        "gen_loss": gen_loss if not args.freeze_backbone else None,
        "pred_loss": pred_loss if args.prediction_module else None,
        "probs": wandb.Histogram(np_histogram=hist) if args.use_wandb else None,
    }


def main(args):

    if args.use_wandb:
        import wandb
        wandb.init(
        #     project=args.wandb_project,
        #     entity=args.wandb_entity,
        #     name=args.wandb_run_name,
        )
        wandb.config.update(args)


    if args.seed:
        set_all_seeds(args.seed, verbose=True)

    # load config, tokenizer, model
    MODEL_NAME = "gogamza/kobart-summarization"
    config = BartConfig.from_pretrained(MODEL_NAME)
    tokenizer = BartTokenizerFast.from_pretrained(MODEL_NAME)
    model = getattr(models, args.model_arch).from_pretrained(MODEL_NAME)

    # freeze backbone
    if args.freeze_backbone:
        print("== Frozen Layers =================================================")
        print(freeze(model, ["model", "lm_head"], exact=False))
        print("====================================================================")

    wandb.watch(model, log='all', log_freq=500)  

    train_dataset = SummaryDataset(args.train_path, tokenizer, is_train=True) if args.do_train else None
    eval_dataset  = SummaryDataset(args.eval_path, tokenizer, is_train=True) if args.do_eval or args.do_predict else None

    if train_dataset is not None:
        print(f"train_dataset length: {len(train_dataset)}")
    if eval_dataset is not None:
        print(f"eval_dataset length: {len(eval_dataset)}")

    train_dl = DataLoader(
        train_dataset, 
        args.per_device_train_batch_size, 
        shuffle=True, 
        collate_fn=lambda x: collate_fn(x, pad_token_idx=tokenizer.pad_token_id),
    ) if args.do_train else None

    eval_dl = DataLoader(
        train_dataset if eval_dataset is None else eval_dataset, 
        args.per_device_eval_batch_size, 
        shuffle=False, 
        collate_fn=lambda x: collate_fn(x, pad_token_idx=tokenizer.pad_token_id),
    ) if args.do_eval or args.do_predict else None

    # optimizer
    # TODO: LR scheduler
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay,
        betas=[args.adam_beta1, args.adam_beta2],
    )

    # train loop
    if not args.no_cuda:
        device = torch.device("cpu") if args.no_cuda or not torch.cuda.is_available() else torch.device("cuda")
        model.to(device)
    model.train()

    total_steps = 0
    optimizer.zero_grad()

    if args.do_train:
        for epoch in range(int(args.num_train_epochs)):
            print("=" * 10 + "Epoch " + str(epoch+1) + " has started! " + "=" * 10)
            total_steps = train_loop(args, model, tokenizer, train_dl, eval_dl, optimizer, total_steps)

            # save the trained model at the end of every epoch
            model.save_pretrained(os.path.join(args.output_dir, f"epoch_{epoch}"))
            
            if args.do_predict:
                print("=" * 10 + "Epoch " + str(epoch+1) + " predict has started! " + "=" * 10)
                pred, _ = predict(args, model, eval_dl, tokenizer)
                with open(os.path.join(args.output_dir, f"pred_epoch_{epoch}.json"), 'w', encoding="utf-8") as f:
                    json.dump(pred, f, ensure_ascii=False)        
    
    # At the end of the whole training,
    # the final evaluation and prediction loop will run!
    if args.do_eval:
        print("=" * 10 + "The final evaluation loop has started!" + "=" * 10)
        eval(args, model, tokenizer, eval_dl, total_steps)

    if args.do_predict:
        print("=" * 10 + "The final prediction loop has started!" + "=" * 10)
        pred_sents, _ = predict(args, model, eval_dl, tokenizer)
        with open(os.path.join(args.output_dir, f"pred_final.json"), 'w', encoding="utf-8") as f:
            json.dump(pred_sents, f, ensure_ascii=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train model.")
    parser = add_train_args(parser)
    parser = add_predict_args(parser)
    parser = add_wandb_args(parser)
    
    args = parser.parse_args()
    
    main(args)