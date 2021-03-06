import os
import json
import argparse
from typing import Optional, List

import pandas as pd
import pyarrow.parquet as pq

import torch
from torch.utils.data import DataLoader

from transformers import BartTokenizerFast, BartForConditionalGeneration

from arguments import add_inference_args, add_predict_args
import models
from dataset import SummaryDataset
from utils import collate_fn, compute_metrics, get_eos_positions
from truncate import batch_truncate_with_eq, gather_lengths, concat_sentences

from tqdm import tqdm


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

def generate_summary(args, model, batch, device):

    summary_ids = None
    if args.generate_method == "greedy":
        summary_ids = model.generate(
            input_ids=batch["input_ids"].to(device), 
            attention_mask=batch["attention_mask"].to(device),  
            max_length=args.max_length, 
            min_length=args.min_length,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            length_penalty=args.length_penalty,
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
            length_penalty=args.length_penalty,
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
            length_penalty=args.length_penalty,
            top_k=50,
            top_p=0.92,
        )

    return summary_ids

def simple_extraction(args, model, batch, tokenizer, device):
    input_ids = batch["input_ids"].clone().to(device)  # (B, L_src)
    attention_mask = batch["attention_mask"].clone().to(device)  # (B, L_src)

    ext_out = model.classify(input_ids=input_ids, attention_mask=attention_mask)

    top_ext_ids = get_top_k_sentences(
        logits=ext_out.logits.clone().detach().cpu(), 
        eos_positions=batch["eos_positions"], 
        k = args.top_k,
    )
    gen_batch = extract_sentences(batch["input_ids"], batch["eos_positions"], top_ext_ids, tokenizer)

    return gen_batch, top_ext_ids

def recursive_extraction(args, model, batch, tokenizer, device):

    input_ids = batch["input_ids"]

    while input_ids.size(1) > 0:

        _input_ids, input_ids = batch_truncate_with_eq(
            input_ids, 
            model.config.max_position_embeddings - model.config.extra_pos_embeddings, 
            sep=tokenizer.eos_token_id, 
            padding_value=tokenizer.pad_token_id, 
            eos_value=tokenizer.eos_token_id, 
            return_mapping=False,
            overflow=False,
        )

        lengths = gather_lengths(_input_ids, tokenizer.pad_token_id)

        _attention_mask = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor([1.0] * l) for l in lengths], 
            batch_first=True, 
            padding_value=0.0
        )

        _input_ids_c = _input_ids.to(device)
        _attention_mask_c = _attention_mask.to(device)

        ext_out = model.classify(
            input_ids=_input_ids_c, 
            attention_mask=_attention_mask_c,
        )

        _eos_positions = get_eos_positions(_input_ids, tokenizer)
        
        top_ext_ids = get_top_k_sentences(
            logits=ext_out.logits.clone().detach().cpu(),
            eos_positions=_eos_positions,
            k=args.top_k,
        )
        _ext_batch = extract_sentences(_input_ids, _eos_positions, top_ext_ids, tokenizer)
        
        if input_ids.size(1) > 0:
            input_ids = concat_sentences(_ext_batch["input_ids"], input_ids, tokenizer.pad_token_id)
            continue
        else:
            return _ext_batch, top_ext_ids

def predict(args, model, test_dl, tokenizer) -> List[str]:

    device = torch.device("cpu") if args.no_cuda or not torch.cuda.is_available() else torch.device("cuda")

    model.to(device)
    model.eval()
    
    pred_sentences = []
    pred_ext_ids = []

    with torch.no_grad():
        for batch in tqdm(test_dl):
            if args.extractive:
                if args.classify_method == "simple":
                    batch, top_ext_ids = simple_extraction(args, model, batch, tokenizer, device)
                elif args.classify_method == "recursive":
                    batch, top_ext_ids = recursive_extraction(args, model, batch, tokenizer, device)

            summary_ids = generate_summary(args, model, batch, device)
        
            summary_sent = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
            pred_sentences.extend(summary_sent)

            if args.extractive:
                # remove invalid ids for highlighting
                top_ext_ids = top_ext_ids.tolist()
                valid_ext_ids = []
                for i in range(len(top_ext_ids)):
                    valid_ext_ids.append([id for id in top_ext_ids[i] if id >= 0])
                pred_ext_ids.extend(valid_ext_ids)

    return pred_sentences, pred_ext_ids


def _get_ref_sentences(reference_file):
    file_ext = os.path.splitext(reference_file)[-1].lower()
    if file_ext == ".parquet":
        ref_df = pq.read_table(reference_file, columns=["abstractive"])
        return ref_df["abstractive"].to_pylist()
    elif file_ext == ".json":
        ref_df = pd.read_json(reference_file)
        return ref_df["abstractive"].tolist()


def main(args):
    # tokenizer, model
    tokenizer = BartTokenizerFast.from_pretrained(args.tokenizer)
    try:
        # load saved model
        with open(os.path.join(args.model, "config.json"), "r") as f:
            architecture = json.load(f)["architectures"][0]
        model = getattr(models, architecture).from_pretrained(args.model)
        assert args.extractive == True
        print("Loaded a custom model.")
    except FileNotFoundError:
        # load from huggingface
        model = BartForConditionalGeneration.from_pretrained(args.model)
        assert args.extractive == False
        print("Loaded a pretrained model from Huggingface.")
    
    # get data
    OUTPUT_DIR = "./outputs/summary_outputs"
    save_file_name = "summary_output.json"

    if args.save_json_name is not None:
        assert os.path.splitext(args.save_json_name)[-1] == ".json", "save_json_name must end with '.json'"
        save_file_name = args.save_json_name

    if os.path.isfile(os.path.join(OUTPUT_DIR, save_file_name)) and not args.overwrite:
        print(f'{save_file_name} has already been generated.')
        return

    test_dataset = SummaryDataset(
        args.test_file_path,
        tokenizer,
        truncate = True if args.classify_method == "simple" else False
    )

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
    
    if args.compute_metrics:
        ref_sents = _get_ref_sentences(args.test_file_path)
        print("="*30)
        print("Rouge Scores:\n", compute_metrics(pred_sents, ref_sents, args.apply_none))
        print("="*30)

    test_title = test_dataset.get_title_column()
    test_text = test_dataset.get_text_column()

    output = []
    for i, id in enumerate(test_id):
        output.append({
            "id": id,
            "title": test_title[i],
            "text": test_text[i],
            "extract_ids": pred_ext_ids[i] if args.extractive else None,
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
