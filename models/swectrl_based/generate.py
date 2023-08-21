import os
import re
import json
import argparse
import math
import dataclasses as dc
from operator import itemgetter
from pathlib import Path
from pprint import pprint

import numpy as np
import torch
import transformers

import jsonlines as jsl
from tqdm import tqdm

from util import FineTuningArguments, add_task_special_tokens
from dataset import (
    SweQuadMCDataset, TokenizeTransform,
    CTRLAregLeftToRightTransform, TextualDatasetFromYAML,
    TextualDatasetFromJsonLines
)

def print_gen_step(idx):
    wlen = 20
    if idx >= 0:
        print("{0} GENERATION STEP {1} {0}".format("="*wlen, idx))
    else:
        print("{0} GENERATION FINISHED {0}".format("="*wlen))

def print_prompt(pr):
    print("\n\tPROMPT: {}\n".format(pr))

def print_generated(txt):
    print("\nGENERATED: {}\n".format(txt))

def print_generated_parts(parts, tok):
    total_text = ""

    Np = len(parts)
    for i, p in enumerate(parts):
        txt_ids, s_id, s_len = p
        total_text += tok.batch_decode(txt_ids[:,s_id:s_id+s_len])[0]
        if i != Np - 1:
            total_text += " >|< "

    print("\nGENERATED: {}\n".format(total_text))


def add_control_code(text, text_ids, control, **kwargs):
    print_prompt("{} {}".format(text, control))
    cc_id = tokenizer(control, return_tensors='pt').input_ids.to(device)
    new_prompt = torch.cat((text_ids, cc_id), dim=1)
    outputs = model.generate(new_prompt, **kwargs)
    return tokenizer.batch_decode(outputs)[0], outputs


def parse_choices(text):
    parts = re.split("(?<= )([a-z]\))(?= )", text)
    choices = []
    i, cum = 0, []
    for p in parts:
        if p:
            cum.append(p)
            if i > 0 and i % 2 == 0:
                choices.append(" ".join(cum).strip())
                cum = []
            i += 1
    if cum:
        choices.append(" ".join(cum))
    return choices


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore") 

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, required=True, help="Checkpoint path")
    parser.add_argument('-l', '--loader', type=str, help="Which dataset to load")
    parser.add_argument('-d', '--dataset', type=str, help="Dataset path")
    parser.add_argument('-m', '--max-datapoints', type=int, default=-1, help="Maximum number of datapoints to take")
    parser.add_argument('-c', '--control', type=str, help="Control code")
    parser.add_argument('-p', '--prompt', type=str, help="Prompt (optional)")
    parser.add_argument('-o', '--output', type=str, help="Suffix for the output file")
    parser.add_argument('-fa', '--ft_args_file', type=str, required=True, help="Fine-tuning settings file")
    parser.add_argument('-n', '--nsamples', default=-1, type=int)
    parser.add_argument('-numo', '--num_only', action='store_true', help="Only return the number of requested MCQs per text")
    parser.add_argument('-gd', '--generate-deterministic', action='store_true', help="Whether to generate in a deterministic fashion")
    args = parser.parse_args()

    print("Torch version: {}".format(torch.__version__))
    print("HF Transformers version: {}".format(transformers.__version__))
    
    torch.set_grad_enabled(False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ft_args = torch.load(args.ft_args_file)
    
    swectrl_model_key = "dkalpakchi/SweCTRL-Mini"
    tokenizer = transformers.AutoTokenizer.from_pretrained(swectrl_model_key)
    tokenizer, START_C_CODES, END_C_CODES = add_task_special_tokens(tokenizer)
    
    eos_token_id = tokenizer.convert_tokens_to_ids(END_C_CODES["mcq"])

    if args.generate_deterministic:
        generation_kwargs = {
            "do_sample": False
        }
    else:
        generation_kwargs = {
            "do_sample": True,
            "top_p": 0.9
        }

    generation_kwargs['pad_token_id'] = tokenizer.pad_token_id
    generation_kwargs['max_new_tokens'] = 60
    if eos_token_id:
        generation_kwargs['eos_token_id'] = eos_token_id

    model = transformers.CTRLLMHeadModel.from_pretrained(args.file, local_files_only=True)
    
    # print(model)
    print("Loaded a model with {} parameters".format(
        sum(p.numel() for p in model.parameters() if p.requires_grad)
    ))
    model.to(device)
    model.eval()

    if args.prompt:
        text = ""
        if args.control:
            text = START_C_CODES[args.control]
        if text:
            text = "{} {}".format(text, args.prompt)
        else:
            text = args.prompt
        text += " {}".format(START_C_CODES["mcq"])
        
        if ft_args.q_prefix:
            text += " {}".format(ft_args.q_prefix)
        
        input_ids = tokenizer(text, return_tensors='pt').input_ids.to(device)
        
        LIMITS = {
            'prompt': input_ids.shape[1],
            'default': ft_args.max_sequence_length - 1,
        }
        total_length = LIMITS['prompt'] + args.length
        max_len = min(total_length, LIMITS['default'])
        max_new_tokens = max_len - LIMITS['prompt']
        
        output = model.generate(
            input_ids, max_new_tokens=max_new_tokens,
            early_stopping=True, **generation_kwargs
        )
        decoded_text = tokenizer.batch_decode(output)[0]
        _, _, mcq = decoded_text.partition(START_C_CODES['mcq'])
    else:
        # Dataset
        if args.loader == 'swequad-mc':
            ds = SweQuadMCDataset(args.dataset.split(","))
        elif args.loader == 'plugga':
            ds = TextualDatasetFromJsonLines(args.dataset.split(","))
        else:
            ds = TextualDatasetFromYAML(args.dataset.split(","))
        
        max_ctx_len = 192
        max_seq_len = 256
        
        ds = TokenizeTransform(
            ds, tokenizer, return_ids=True,
            for_generation=True,
            q_prefix=ft_args.q_prefix,
            for_model=ft_args.base_model,
            max_context_length=max_ctx_len,
            max_sequence_length=max_seq_len,
            start_codes=START_C_CODES,
            end_codes=END_C_CODES
        )
        
        dev_ds = CTRLAregLeftToRightTransform(ds, for_generation=True)
        
        if args.output:
            out_fname = "{}_{}.jsonl".format(args.file.strip("/"), args.output)
        else:
            out_fname = "{}.jsonl".format(args.file.strip("/"))
        with jsl.open(out_fname, 'w') as writer:
            specs = {}
            specs.update(dc.asdict(ft_args))
            specs.update(generation_kwargs)
            writer.write(specs)
            part = dev_ds if args.max_datapoints <= 0 else dev_ds[:args.max_datapoints]
            for dp in tqdm(part):
                res = {
                    'text': tokenizer.decode(dp["input_ids"]),
                    'mcqs': []
                }
                
                if args.nsamples > 0:
                    num_mcqs = args.nsamples
                else:
                    num_mcqs = math.ceil(len(res['text']) / (12.78 * 4.81))
                
                res['req_mcqs'] = num_mcqs

                if args.num_only:
                    writer.write(res)
                    continue

                input_ids = torch.tensor([dp['input_ids']]).to(device)
                for _ in range(num_mcqs):
                    output = model.generate(
                        input_ids,
                        **generation_kwargs
                    )
                    decoded_text = tokenizer.batch_decode(output)[0]
                    _, _, mcq = decoded_text.partition(START_C_CODES['mcq'])
                    mcq = mcq.replace(" )", ")").replace("Fråga :", ft_args.q_prefix)
                    errors = []
                    if "?" not in mcq:
                        errors.append("StemNoQmark")
                    q, _, choices = mcq.partition("?")
                    has_end_code = False
                    if END_C_CODES['mcq'] in choices:
                        choices, _ = choices.split(END_C_CODES['mcq'])
                        has_end_code = True
                    res['mcqs'].append({
                        "stem": "{}?".format(q.replace(ft_args.q_prefix, "").strip()),
                        'choices': parse_choices(choices),
                        "has_end_code": has_end_code,
                        "errors": errors
                    })
                writer.write(res)    


