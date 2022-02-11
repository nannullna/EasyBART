import os
import json
import argparse
from typing import Optional, List

import pandas as pd

import torch
from torch.utils.data import DataLoader

from transformers import BartTokenizerFast, BartForConditionalGeneration

from arguments import add_inference_args, add_predict_args
from models import BartSummaryModelV2
from dataset import SummaryDataset
from utils import collate_fn

from tqdm import tqdm
import glob


def get_top_k_sentences(logits: torch.FloatTensor, eos_positions: torch.LongTensor, k: int = 3):
    returned_tensor = []
    top_ext_ids = torch.argsort(logits, dim=-1, descending=True)
    num_sentences = torch.sum(torch.gt(eos_positions, 0), dim=-1, dtype=torch.long)

    for i in range(len(top_ext_ids)):
        top_ext_id = top_ext_ids[i]
        top_ext_id = top_ext_id[top_ext_id < num_sentences[i]]
        top_ext_id = top_ext_id[:k]
        top_k, _ = torch.sort(top_ext_id)

        padding = torch.tensor([-1] * k)
        top_k = torch.cat([top_k, padding])[:k]

        returned_tensor.append(top_k.unsqueeze(0))
    
    returned_tensor = torch.cat(returned_tensor, dim=0)

    return returned_tensor


def extract_sentences(
    input_ids: torch.FloatTensor,
    eos_positions: torch.LongTensor,
    ext_ids: torch.LongTensor,
    tokenizer: BartTokenizerFast,  
):
    PAD = tokenizer.pad_token_id
    gen_batch_inputs = []
    attention_mask = []

    for i in range(input_ids.size(0)):
        ids = ext_ids[i][ext_ids[i] >= 0].tolist()
        sentences = [torch.tensor([tokenizer.bos_token_id])]
        for idx in ids:
            from_pos = 1 if idx == 0 else (eos_positions[i, idx-1].item() + 1)
            to_pos = (eos_positions[i, idx].item() + 1)
            
            ext_sentence = input_ids[i, from_pos:to_pos].clone().detach()
            sentences.append(ext_sentence)
        sentences = torch.cat(sentences, dim=0)
        gen_batch_inputs.append(sentences)
        attention_mask.append(torch.ones(len(sentences)))

    gen_batch_inputs = torch.nn.utils.rnn.pad_sequence(gen_batch_inputs, padding_value=PAD, batch_first=True)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, padding_value=0, batch_first=True)

    return {
        "input_ids": gen_batch_inputs,
        "attention_mask": attention_mask,
    }


def predict(args, model, test_dl, tokenizer) -> List[str]:

    device = torch.device("cpu") if args.no_cuda or not torch.cuda.is_available() else torch.device("cuda")

    model.to(device)
    model.eval()
    
    pred_sentences = []
    pred_ext_ids = []

    with torch.no_grad():
        for batch in tqdm(test_dl):
            if not args.pretrained:
                input_ids = batch["input_ids"].clone().to(device)  # (B, L_src)
                attention_mask = batch["attention_mask"].clone().to(device)  # (B, L_src)

                ext_out = model.classify(input_ids=input_ids, attention_mask=attention_mask)

                # TODO: use different k values
                # TODO: implement different criteria (such as probability)!
                top_ext_ids = get_top_k_sentences(
                    logits=ext_out.logits.clone().detach().cpu(), 
                    eos_positions=batch["eos_positions"], 
                    k = args.top_k,
                )
                batch = extract_sentences(batch["input_ids"], batch["eos_positions"], top_ext_ids, tokenizer)

            summary_ids = None
            if args.generate_method == "greedy":
                summary_ids = model.generate(
                    input_ids=batch["input_ids"].to(device), 
                    attention_mask=batch["attention_mask"].to(device),  
                    max_length=args.max_length, 
                    min_length=args.min_length,
                    repetition_penalty=args.repetition_penalty,
                    no_repeat_ngram_size=args.no_repeat_ngram_size,
                )
            elif args.generate_method == "beam":
                summary_ids = model.generate(
                    input_ids=batch["input_ids"].to(device), 
                    attention_mask=batch["attention_mask"].to(device), 
                    num_beams=args.num_beams, 
                    max_length=args.max_length, 
                    min_length=args.min_length,
                    repetition_penalty=args.repetition_penalty,
                    no_repeat_ngram_size=args.no_repeat_ngram_size,
                )
            elif args.generate_method == "sampling":
                summary_ids = model.generate(
                    input_ids=batch["input_ids"].to(device), 
                    attention_mask=batch["attention_mask"].to(device), 
                    do_sample=True,
                    max_length=args.max_length, 
                    min_length=args.min_length,
                    repetition_penalty=args.repetition_penalty,
                    no_repeat_ngram_size=args.no_repeat_ngram_size,
                    top_k=50,
                    top_p=0.92,
                )
            
            summary_sent = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
            pred_sentences.extend(summary_sent)

            if not args.pretrained:
                # remove invalid ids for highlighting
                top_ext_ids = top_ext_ids.tolist()
                valid_ext_ids = []
                for i in range(len(top_ext_ids)):
                    valid_ext_ids.append([id for id in top_ext_ids[i] if id >= 0])
                pred_ext_ids.extend(valid_ext_ids)

    return pred_sentences, pred_ext_ids


def main(args):
    # tokenizer, model
    tokenizer = BartTokenizerFast.from_pretrained(args.tokenizer)
    if args.pretrained:
        model = BartForConditionalGeneration.from_pretrained(args.model)
    else:
        model = BartSummaryModelV2.from_pretrained(args.model)
    
    # get data
    OUTPUT_DIR = "./outputs"
    save_file_name = "summary_output.json"

    if args.save_json_name is not None:
        assert os.path.splitext(args.save_json_name)[-1] == ".json", "save_json_name must end with '.json'"
        save_file_name = args.save_json_name

    if os.path.isfile(os.path.join(OUTPUT_DIR, save_file_name)) and not args.overwrite:
        print(f'{save_file_name} has already been generated.')
        return

    test_dataset = SummaryDataset(args.test_file_path, tokenizer)

    print(f"test dataset length: {len(test_dataset)}")
    
    test_dl = DataLoader(
        test_dataset, 
        args.per_device_eval_batch_size, 
        shuffle=False, 
        collate_fn=lambda x: collate_fn(x, pad_token_idx=tokenizer.pad_token_id, sort_by_length=False),
        drop_last=False,
    )

    pred_sents, pred_ext_ids = predict(args, model, test_dl, tokenizer)
            
    print("Inference completed!")
    test_id = test_dataset.get_id_column()
    
    assert len(test_id) == len(pred_sents), "lengths of test_id and pred_sents do not match"
    
    test_title = test_dataset.get_title_column()

    output = []
    for i, id in enumerate(test_id):
        output.append({
            "id": id,
            "title": test_title[i],
            "extract_ids": pred_ext_ids[i] if not args.pretrained else None,
            "summary": pred_sents[i]
        })

    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    with open(os.path.join(OUTPUT_DIR, save_file_name), 'w', encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser = add_inference_args(parser)
    parser = add_predict_args(parser)

    args = parser.parse_args()

    main(args)
