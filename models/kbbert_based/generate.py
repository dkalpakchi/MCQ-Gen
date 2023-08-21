import json
import math
import copy
import sys
import dataclasses as dc
import argparse
import os
import mmap
import string
from operator import itemgetter
from collections import defaultdict, OrderedDict
from pprint import pprint
from io import BytesIO

import torch
import torch.nn.functional as F
import numpy as np
from scipy.special import softmax
from transformers import (
    AutoTokenizer, BertForMaskedLM, BertConfig
)
from transformers.modeling_outputs import MaskedLMOutput

import jsonlines as jsl
from tqdm import tqdm

from dataset import (
    SweQuadMCDataset, TokenizeTransform,
    BERTAregLeftToRightTransform,
    BERTAregArbitraryOrderTransform,
    TextualDatasetFromYAML,
    TextualDatasetFromJsonLines
)
from common import KBBERT_HF_KEY, AREG_ARB, AREG_LTR, BLANK_TOKEN


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class Cache:
    def __init__(self, fname):
        # Don't forget to close both when done with them
        if not os.path.exists(fname):
            size = 72*1024*1024*1024 # 72GB in bytes
            multiplier = 18*1024*1024*1024
            f = open(fname, mode="wb")
            for _ in range(size//multiplier):
                f.write(multiplier*b'\0')
            f.close()
        self.__file_obj = open(fname, mode="r+b")
        self.__mmap_obj = mmap.mmap(self.__file_obj.fileno(), length=0, access=mmap.ACCESS_WRITE)
        self.__seekmap = {}
        self.__ptr = 0

    def put(self, key, matrix, compressed=False):
        self.__mmap_obj.seek(self.__ptr)
        np_io = BytesIO()
        if compressed:
            np.savez_compressed(np_io, matrix, allow_pickle=True)
        else:
            np.save(np_io, matrix, allow_pickle=True)
        np_bytes = np_io.getvalue()
        self.__mmap_obj.write(np_bytes)
        N_bytes = len(np_bytes)
        self.__seekmap[key] = (self.__ptr, N_bytes) # (where to seek, how many bytes to read)
        self.__ptr += N_bytes

    def get(self, key, compressed=False):
        ptr, N_bytes = self.__seekmap[key]
        self.__mmap_obj.seek(ptr)
        read_bytes = self.__mmap_obj.read(N_bytes+1)
        res = np.load(BytesIO(read_bytes), allow_pickle=True)
        if compressed:
            return res['arr_0']
        else:
            return res

    def __contains__(self, key):
        return key in self.__seekmap


class Generator:
    def __init__(self, tokenizer, model, device='cpu', generation_kwargs=None):
        self.__tok = tokenizer
        self.__m = model
        self.__device = device
        #self.__collator = GenDataCollatorForTokenClassification(self.__tok)
        self.__gen_kwargs = generation_kwargs
        self.__m.eval()
        self.__m.to(device)

    @property
    def tok(self):
        return self.__tok

    def dict2tensors(self, d):
        for k, v in d.items():
            if torch.is_tensor(v):
                d[k] = v.to(self.__device) # means we already have a batch dimension
            else:
                d[k] = torch.tensor([v]).to(self.__device) # adding batch dimension
        return d

    def dp2str(self, l, dp_id):
        return "{}->{}".format(dp_id, ";".join(map(str, l)))

    def filter_dp(self, dp, to_delete):
        if to_delete:
            dp["input_ids"] = [x for i, x in enumerate(dp["input_ids"]) if i not in to_delete]
            dp["token_type_ids"] = [x for i, x in enumerate(dp["token_type_ids"]) if i not in to_delete]
            dp["attention_mask"] = [x for i, x in enumerate(dp["attention_mask"]) if i not in to_delete]
        return dp

    def convert_dp(self, datapoint, append_mask=True):
        if isinstance(datapoint, str):
            dp = self.__tok(datapoint)
            if dp['input_ids'][-1] != self.__tok.mask_token_id and append_mask:
                dp['input_ids'].append(self.__tok.mask_token_id)
                dp['token_type_ids'].append(1)
                dp['attention_mask'].append(1)
            return dp
        else:
            return datapoint

    def generate_areg_ltr(self, datapoint):
        dp_c = self.convert_dp(datapoint)
        dp = copy.deepcopy(dp_c)
        real_stem = dp.pop("stem") if "stem" in dp else None
        real_key = dp.pop("key") if "key" in dp else None
        real_distractors = dp.pop("distractors") if "distractors" in dp else None
        
        num_alternatives = self.__gen_kwargs['num_alternatives']
        stem_hard_limit = self.__gen_kwargs['stem_hard_limit']
        alt_hard_limit = self.__gen_kwargs['alt_hard_limit']
        do_sample = self.__gen_kwargs['do_sample']

        errors = []
        gen_stem, gen_choices = [], []
        num_gen = -1
        while num_gen < num_alternatives:
            alt = []

            try:
                outputs = self.__m(**self.dict2tensors(dict(dp)))
            except RuntimeError as e:
                errors.append(str(e))
                break

            logits = outputs.logits.cpu().detach().numpy()
            
            if do_sample:
                idx_list = list(range(logits.shape[-1]))
                tok_id = np.random.choice(idx_list, p=softmax(logits[0][-1]))
            else:
                tok_id = np.argmax(logits[0][-1]) # get the max logit for the last word
            alt.append(tok_id)
            hard_limit = stem_hard_limit if num_gen < 0 else alt_hard_limit
            while tok_id != self.__tok.sep_token_id and len(alt) <= hard_limit:
                dp["input_ids"].insert(-1, tok_id)
                dp["token_type_ids"].append(1)
                dp["attention_mask"].append(1)

                try:
                    outputs = self.__m(**self.dict2tensors(dict(dp)))
                except RuntimeError as e:
                    errors.append(str(e))
                    break
                logits = outputs.logits.cpu().detach().numpy()
                if do_sample:
                    idx_list = list(range(logits.shape[-1]))
                    tok_id = np.random.choice(idx_list, p=softmax(logits[0][-1]))
                else:
                    tok_id = np.argmax(logits[0][-1]) # get the max logit for the last word

                alt.append(tok_id)
            dp["input_ids"].insert(-1, tok_id)
            dp["token_type_ids"].append(1)
            dp["attention_mask"].append(1)

            if alt[-1] != self.__tok.sep_token_id:
                dp["input_ids"].insert(-1, self.__tok.sep_token_id)
                dp["token_type_ids"].append(1)
                dp["attention_mask"].append(1)

            if num_gen < 0:
                gen_stem = alt
            else:
                gen_choices.append(alt)
            alt = []
            num_gen += 1

            if len(errors) > 0:
                break

        return {
            "stem": {
                "g": self.__tok.decode(gen_stem),
                "r": self.__tok.decode(real_stem) if real_stem else None
            },
            "choices": {
                "g": [self.__tok.decode(a) for a in gen_choices],
                "r": [self.__tok.decode(a) for a in [real_key] + real_distractors] if real_key and real_distractors else None
            },
            "errors": errors
        }

    def generate_areg_arb(self, datapoint, criterion='max_prob', all_at_once=False):
        def add_masked(dx, length):
            for x in range(length):
                dx["input_ids"].append(self.__tok.mask_token_id)
                dx["token_type_ids"].append(1)
                dx["attention_mask"].append(1)
            return dx

        def add_sep(dx):
            dx["input_ids"].append(self.__tok.sep_token_id)
            dx["token_type_ids"].append(1)
            dx["attention_mask"].append(1)
            return dx

        def replace_masked(dx, lst):
            rep = 0
            for i, x in enumerate(dx["input_ids"]):
                if x == self.__tok.mask_token_id:
                    dx["input_ids"][i] = lst[rep]
                    rep += 1
            return dx

        dp_c = self.convert_dp(datapoint, append_mask=False)
        dp = copy.deepcopy(dp_c) # not to mess up the original dataset

        max_stem_length = 30
        max_alt_length = 20

        # Generate question if model supports it
        real_stem = dp.pop("stem") if "stem" in dp else None
        real_key = dp.pop("key") if "key" in dp else None
        real_distractors = dp.pop("distractors") if "distractors" in dp else None
        
        num_alternatives = self.__gen_kwargs['num_alternatives']
        do_sample = self.__gen_kwargs['do_sample']

        gen_stem, gen_choices = [], []
        errors = []

        if all_at_once:
            dp = add_masked(dp, max_stem_length)
            dp = add_sep(dp)
            for idx in range(num_alternatives):
                dp = add_masked(dp, max_alt_length)
                if idx < num_alternatives - 1:
                    dp = add_sep(dp)
            gen_length = max_stem_length + num_alternatives * (max_alt_length + 1)
            mcq = [-1] * gen_length  
            mask = [False] * gen_length
            for i in range(num_alternatives):
                mcq[max_stem_length + i * max_alt_length] = self.__tok.sep_token_id
                mask[max_stem_length + i * max_alt_length] = True
            
            while not all(mask):
                try:
                    outputs = self.__m(**self.dict2tensors(dict(dp)))
                except RuntimeError as e:
                    errors.append(str(e))
                    break
                logits = outputs.logits.cpu().detach().numpy()
                mcq_logits = logits[0][-gen_length:]
                mcq_logits[mask] = -100

                if criterion == 'min_ent':
                    probs = softmax(mcq_logits, axis=-1)
                    entropy = -(probs * np.log(probs + 1e-15)).sum(axis=1)
                    entropy[mask] = float('inf')
                    d_pos = np.argmin(entropy)
                    tok_id = np.argmax(mcq_logits[d_pos]) # get the max logit for the whole distractor

                    #top5 = np.argsort(dis_logits[d_pos])[-5:]
                    #print("==> Contenders")
                    #for j in range(1, 6):
                    #    print(top5[-j], self.__tok.decode([top5[-j]]), probs[d_pos][top5[-j]])
                else:
                    if do_sample:
                        d_pos, _ = np.unravel_index(np.argmax(mcq_logits), mcq_logits.shape)
                        
                        idx_list = list(range(mcq_logits.shape[-1]))
                        tok_id = np.random.choice(idx_list, p=softmax(mcq_logits[d_pos]))
                    else:
                        d_pos, tok_id = np.unravel_index(np.argmax(mcq_logits), mcq_logits.shape) # get the max logit for the whole distractor

                    #probs = softmax(alt_logits[d_pos])
                    #top5 = np.argsort(dis_logits[d_pos])[-5:]
                    #print("==> Contenders")
                    #for j in range(1, 6):
                    #    print(top5[-j], self.__tok.decode([top5[-j]]), probs[top5[-j]])

                dp["input_ids"][-gen_length + d_pos] = tok_id
                mcq[d_pos] = tok_id
                mask[d_pos] = True

            gen_stem = mcq[:max_stem_length + 1]
            gen_choices = []
            for i in range(num_alternatives):
                alt = mcq[max_stem_length + 1 + i * max_alt_length:max_stem_length + 1 + (i+1) * max_alt_length]
                gen_choices.append(alt)
        else:
            num_gen = -1
            while num_gen < num_alternatives:
                gen_length = max_stem_length if num_gen < 0 else max_alt_length
                dp = add_masked(dp, gen_length)

                alt = [-1] * gen_length
                mask = [False] * gen_length
                while not all(mask):
                    try:
                        outputs = self.__m(**self.dict2tensors(dict(dp)))
                    except RuntimeError as e:
                        errors.append(str(e))
                        break
                    logits = outputs.logits.cpu().detach().numpy()
                    alt_logits = logits[0][-gen_length:]
                    alt_logits[mask] = -100

                    if criterion == 'min_ent':
                        probs = softmax(alt_logits, axis=-1)
                        entropy = -(probs * np.log(probs + 1e-15)).sum(axis=1)
                        entropy[mask] = float('inf')
                        d_pos = np.argmin(entropy)
                        tok_id = np.argmax(alt_logits[d_pos]) # get the max logit for the whole distractor

                        #top5 = np.argsort(dis_logits[d_pos])[-5:]
                        #print("==> Contenders")
                        #for j in range(1, 6):
                        #    print(top5[-j], self.__tok.decode([top5[-j]]), probs[d_pos][top5[-j]])
                    else:
                        if do_sample:
                            d_pos, _ = np.unravel_index(np.argmax(alt_logits), alt_logits.shape)
                            
                            idx_list = list(range(alt_logits.shape[-1]))
                            tok_id = np.random.choice(idx_list, p=softmax(alt_logits[d_pos]))
                        else:
                            d_pos, tok_id = np.unravel_index(np.argmax(alt_logits), alt_logits.shape) # get the max logit for the whole distractor

                        #probs = softmax(alt_logits[d_pos])
                        #top5 = np.argsort(dis_logits[d_pos])[-5:]
                        #print("==> Contenders")
                        #for j in range(1, 6):
                        #    print(top5[-j], self.__tok.decode([top5[-j]]), probs[top5[-j]])

                    dp["input_ids"][-gen_length + d_pos] = tok_id
                    alt[d_pos] = tok_id
                    mask[d_pos] = True

                if alt[-1] != self.__tok.sep_token_id:
                    dp["input_ids"].append(self.__tok.sep_token_id)
                    dp["token_type_ids"].append(1)
                    dp["attention_mask"].append(1)

                if num_gen < 0:
                    gen_stem = alt
                else:
                    gen_choices.append(alt)
                alt = []
                num_gen += 1
                if len(errors) > 0:
                    break
                    
        return {
            "stem": {
                "g": self.__tok.decode(gen_stem),
                "r": self.__tok.decode(real_stem) if real_stem else None
            },
            "choices": {
                "g": [self.__tok.decode(a) for a in gen_choices],
                "r": [self.__tok.decode(a) for a in [real_key] + real_distractors] if real_key and real_distractors else None
            },
            "errors": errors
        }

    def generate(self, datapoint, all_at_once=False):
        formulation = self.__gen_kwargs.get('formulation')
        if formulation is None:
            raise NotImplementedError("The formulation {} is not implemented!".format(formulation))
        else:
            gen_method = getattr(self, "generate_{}".format(formulation))
            if formulation == AREG_ARB:
                return gen_method(datapoint, all_at_once=all_at_once)
            else:
                return gen_method(datapoint)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, required=True, help="Checkpoint path")
    parser.add_argument('-fa', '--ft_args_file', type=str, required=True, help="Fine-tuning settings file")
    parser.add_argument('-l', '--loader', type=str, help="Which dataset to load")
    parser.add_argument('-d', '--dataset', type=str, help="Dataset path")
    parser.add_argument('-m', '--max-datapoints', type=int, default=-1, help="Maximum number of datapoints to take")
    parser.add_argument('-p', '--prompt', type=str, help="Prompt (optional)")
    parser.add_argument('-o', '--output', type=str, help="Suffix for the output file")
    parser.add_argument('-numo', '--num_only', action='store_true', help="Only return the number of requested MCQs per text")
    parser.add_argument('-n', '--nsamples', default=-1, type=int)
    parser.add_argument('-gd', '--generate-deterministic', action='store_true', help="Whether to generate in a deterministic fashion")
    args = parser.parse_args()

    torch.set_grad_enabled(False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ft_args = torch.load(args.ft_args_file)
    print(ft_args)

    tokenizer = AutoTokenizer.from_pretrained(KBBERT_HF_KEY)
    #if dg_args.max_question_length > 0:
    #    tok.add_special_tokens({'additional_special_tokens': tok.additional_special_tokens + ['[BLANK]']})
    
    model = BertForMaskedLM.from_pretrained(args.file, local_files_only=True)
    
    if args.generate_deterministic:
        generation_kwargs = {
            "do_sample": False
        }
    else:
        generation_kwargs = {
            "do_sample": True
        }
    generation_kwargs['formulation'] = ft_args.formulation
    generation_kwargs['num_alternatives'] = 4
    generation_kwargs['stem_hard_limit'] = 30
    generation_kwargs['alt_hard_limit'] = 20

    print(generation_kwargs)

    gen = Generator(
        tokenizer, model, device,
        generation_kwargs=generation_kwargs
    )
    
    text = None
    if args.prompt:
        if text:
            text = "{} {} [SEP]".format(text, args.prompt)
        else:
            text = "{} [SEP]".format(args.prompt)
        
        if ft_args.q_prefix:
            text += " {}".format(ft_args.q_prefix)
        print(text)
        print(gen.generate(text, all_at_once=ft_args.all_at_once))
    else:
        # Dataset
        if args.loader == 'swequad-mc':
            ds = SweQuadMCDataset(args.dataset.split(","))
        elif args.loader == 'plugga':
            ds = TextualDatasetFromJsonLines(args.dataset.split(","))
        else:
            ds = TextualDatasetFromYAML(args.dataset.split(","))
        
        max_ctx_len = 441 # ~64 * 1.1 tokens = ~71 tokens for MCQ
        max_seq_len = 512
        
        if ft_args.formulation == AREG_ARB:
            max_ctx_len = 384 # to accommodate fixed lengths of stems and alternatives
            tokenizer.add_special_tokens({
                'additional_special_tokens': tokenizer.additional_special_tokens + [BLANK_TOKEN]
            })
        
        ds = TokenizeTransform(
            ds, tokenizer, return_ids=True,
            for_generation=True,
            q_prefix=ft_args.q_prefix,
            for_model=ft_args.base_model,
            max_context_length=max_ctx_len,
            max_sequence_length=max_seq_len
        )
        
        if ft_args.formulation == AREG_LTR:
            dev_ds = BERTAregLeftToRightTransform(ds, for_generation=True)
        elif ft_args.formulation == AREG_ARB:
            dev_ds = BERTAregArbitraryOrderTransform(
                ds, all_at_once=ft_args.all_at_once,
                for_generation=True
            )
     
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

                for _ in range(num_mcqs):
                    mcq = gen.generate(dp, all_at_once=ft_args.all_at_once)
                    res['mcqs'].append({
                        'stem': mcq['stem']['g'],
                        'choices': mcq['choices']['g'],
                        'errors': mcq.get('errors', [])
                    })
                writer.write(res)

