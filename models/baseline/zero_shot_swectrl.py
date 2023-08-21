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

from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList

from util import FineTuningArguments, add_task_special_tokens
from dataset import (
    SweQuadMCDataset, TokenizeTransform,
    CTRLAregLeftToRightTransform, TextualDatasetFromYAML
)


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


class MultiEosTokensCriteria(StoppingCriteria):
    def __init__(self, eos_tokens):
        self.eos_tokens = eos_tokens

    def __call__(self, input_ids, scores, **kwargs):
        lst_inputs = input_ids[0].tolist()
        for x in self.eos_tokens:
            K = len(x)
            if lst_inputs[-K:] == x:
                return True
        return False

def generate_mcq(prompt):
    text = "{} Fråga:".format(prompt)
    input_ids = tokenizer(text, return_tensors='pt').input_ids.to(device)
   
    generation_kwargs['eos_token_id'] = q_eos_token_id
    output = model.generate(input_ids, **generation_kwargs)
    decoded_text = tokenizer.batch_decode(output)[0]
    parts = decoded_text.split("Fråga :")
    
    if len(parts) > 1:
        q = parts[1].strip()
        text = "{} {} Svar:".format(text, q)
        input_ids = tokenizer(text, return_tensors='pt').input_ids.to(device)
       
        generation_kwargs['eos_token_id'] = a_eos_token_id
        output = model.generate(input_ids, **generation_kwargs)
        decoded_text = tokenizer.batch_decode(output)[0]
        a_parts = decoded_text.split("Svar :")
        
        if len(a_parts) > 1:
            a = a_parts[1].strip()
            
            text = "{} Fråga: {} a) {} b)".format(prompt, q, a)
            input_ids = tokenizer(text, return_tensors='pt').input_ids.to(device)
            
            generation_kwargs['max_new_tokens'] = 90
            generation_kwargs['eos_token_id'] = None
            generation_kwargs['stopping_criteria'] = StoppingCriteriaList([
                MultiEosTokensCriteria([tokenizer.encode("Fråga"), tokenizer.encode('e)')])
            ])
            
            output = model.generate(input_ids, **generation_kwargs)
            decoded_text = tokenizer.batch_decode(output)[0]

            # Cleanup
            decoded_text = re.sub('(Fråga|e\))$', '', decoded_text)
            decoded_text = decoded_text.replace(" )", ")")
            decoded_text = decoded_text.replace("Fråga :", "Fråga:")
            decoded_text = decoded_text.replace(text, "").strip()
            choices = [x.strip() for x in re.split("[a-z]\)", decoded_text) if x.strip()]
            return {
                "stem": q,
                "choices": [a] + choices
            }
    return {}


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore") 

    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--loader', type=str, help="Which dataset to load")
    parser.add_argument('-d', '--dataset', type=str, help="Dataset path")
    parser.add_argument('-m', '--max-datapoints', type=int, default=-1, help="Maximum number of datapoints to take")
    parser.add_argument('-c', '--control', type=str, help="Control code")
    parser.add_argument('-p', '--prompt', type=str, help="Prompt (optional)")
    parser.add_argument('-o', '--output', type=str, help="Suffix for the output file")
    parser.add_argument('-n', '--nsamples', default=-1, type=int)
    parser.add_argument('-gd', '--generate-deterministic', action='store_true', help="Whether to generate in a deterministic fashion")
    args = parser.parse_args()

    print("Torch version: {}".format(torch.__version__))
    print("HF Transformers version: {}".format(transformers.__version__))
    
    torch.set_grad_enabled(False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    swectrl_model_key = "dkalpakchi/SweCTRL-Mini"
    tokenizer = transformers.AutoTokenizer.from_pretrained(swectrl_model_key)
    tokenizer, START_C_CODES, END_C_CODES = add_task_special_tokens(tokenizer)
    
    q_eos_token_id = tokenizer.convert_tokens_to_ids("?")
    a_eos_token_id = tokenizer.convert_tokens_to_ids(".")

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
    generation_kwargs['max_new_tokens'] = 30
    generation_kwargs['early_stopping'] = True

    model = transformers.CTRLLMHeadModel.from_pretrained(swectrl_model_key)
    
    # print(model)
    print("Loaded a model with {} parameters".format(
        sum(p.numel() for p in model.parameters() if p.requires_grad)
    ))
    model.to(device)
    model.eval()

    if args.prompt:
        print("Prompt: {}".format(args.prompt))
        for i in range(args.nsamples):
            print(generate_mcq(args.prompt))
    else:
        # Dataset
        if args.loader == 'swequad-mc':
            ds = SweQuadMCDataset(args.dataset.split(","))
        else:
            ds = TextualDatasetFromYAML(args.dataset.split(","))
        
        if args.output:
            out_fname = "zs_baseline_{}.jsonl".format(args.output)
        else:
            out_fname = "zs_baseline.jsonl"
        with jsl.open(out_fname, 'w') as writer:
            specs = generation_kwargs
            writer.write(specs)
            part = ds if args.max_datapoints <= 0 else ds[:args.max_datapoints]
            for dp in tqdm(part):
                res = {
                    'text': dp['context'],
                    'mcqs': []
                }
                
                if args.nsamples > 0:
                    num_mcqs = args.nsamples
                else:
                    num_mcqs = math.ceil(len(res['text']) / (12.78 * 4.81))

                for _ in range(num_mcqs):
                    res['mcqs'].append(generate_mcq(dp['context']))
                writer.write(res)    


