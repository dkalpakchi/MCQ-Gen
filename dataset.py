import os
import re
import sys
import glob
import gzip
import json
import random
import logging
import multiprocessing as mp
import copy
from collections import defaultdict, Counter
from itertools import permutations
from urllib.parse import urlparse

import yaml
import jsonlines as jsl

import numpy as np

import torch
import torch.distributed as dist
from torch.utils.data import Dataset

from tqdm import tqdm

from common import (
    SWECTRL_MODEL, KBBERT_MODEL,
    SWECTRL_HF_KEY, KBBERT_HF_KEY,
    SWEQUAD_DS, QUASI_DS,
    AREG_LTR, AREG_ARB, BLANK_TOKEN
)
from util import add_task_special_tokens

IGNORE_INDEX = -100


FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


def get_text(c, field):
    return c["extra"][field].replace("”", '"').strip() if "extra" in c and c["extra"] else c["text"].replace("”", '"').strip()


# Adapted from https://stackoverflow.com/questions/312443/how-do-i-split-a-list-into-equally-sized-chunks
def chunkify(lst, n, return_last=False):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
    if len(lst) > n and return_last:
        yield lst[-n:]


def group_units(inputs):
    groups = defaultdict(dict)
    yes_no_units = []
    for item in inputs:
        marker_name = item['marker']['name'].lower()

        if marker_name == 'yes/no question':
            yes_no_units.append(item['unit'])

        if marker_name == 'question':
            groups[item['unit']]['question'] = item['content'].replace("”", '"').strip()
        elif marker_name == 'correct answer':
            groups[item['unit']]['correct'] = item['content'].replace("”", '"').strip()
        elif marker_name == 'distractor':
            if 'distractors' not in groups[item['unit']]:
                groups[item['unit']]['distractors'] = set()    
            groups[item['unit']]['distractors'].add(item['content'].replace("”", '"').strip())

    for x in yes_no_units:
        del groups[x]

    return groups


def get_item(seq, i, default):
    try:
        return seq[i]
    except IndexError:
        return default


class GenericDataset(Dataset):
    def __init__(self):
        self._data = []

    def __getitem__(self, idx):
        return self._data[idx]

    def __len__(self):
        return len(self._data)


class TextualDatasetFromYAML(GenericDataset):
    def __init__(self, fnames, text_key="text", control_code=None):
        super().__init__()
        for fname in fnames:
            datapoints = yaml.load(open(fname), Loader=yaml.CLoader)

            for dp in tqdm(datapoints):
                self._data.append({
                    "context": dp[text_key],
                    "control": control_code
                })


class TextualDatasetFromJsonLines(GenericDataset):
    def __init__(self, fnames, text_key="text", control_code=None):
        super().__init__()
        for fname in fnames:
            with jsl.open(fname) as reader:
                for dp in tqdm(reader):
                    self._data.append({
                        "context": dp[text_key],
                        "control": control_code
                    })


class QuasiDataset(GenericDataset):
    def __init__(self, fnames, control=None):
        super().__init__()

        l2i = {v: i for i, v in enumerate('abcd')}

        for fname in sorted(fnames):
            with open(fname) as f:
                data = json.load(f)

                for x in sorted(data, key=lambda x: x['text']):
                    for item in x["test"]:
                        choices = item["mcq"]["choices"]
                        # remove the letters (a - d)
                        choices = [x[2:].strip() for x in choices]

                        key = item['annotations']["correct"]
                        key_basis = item['annotations']['bases'][key]
                       
                        self._data.append({
                            "context": x["text"],
                            "question": item["mcq"]["stem"].partition(" ")[2],
                            "correct": choices[l2i[key]],
                            "correct_basis": None if key_basis is None else (key_basis['start'], key_basis['end']),
                            "distractors": [
                                c for i, c in enumerate(choices) if i != l2i[key]
                            ],
                            "control": control
                        })


class SweQuadMCDataset(GenericDataset):
    def __init__(self, fnames, control=None):
        super().__init__()
        for fname in sorted(fnames):
            d = json.load(open(fname))
            assert 'data' in d, "Wrong format"
            for dp in tqdm(sorted(d["data"], key=lambda x: x["question"])):
                self._data.append({
                    "context": dp["context"].replace("”", '"').strip(), 
                    "question": dp["question"].replace("”", '"').strip(),
                    "correct": None,
                    "correct_basis": None,
                    "distractors": [],
                    "control": control
                })
                for c in dp["choices"]:
                    if c["type"] == "Correct answer":
                        self._data[-1]["correct"] = get_text(c, "comment")
                        self._data[-1]["correct_basis"] = (c['start'], c['end'])
                    elif c["type"] == "Distractor":
                        self._data[-1]["distractors"].append(get_text(c, "comment"))


class CombinedDataset(GenericDataset):
    def __init__(self, *ds):
        super().__init__()

        for d in ds:
            self._data.extend(d._data)


class TokenizeTransform(GenericDataset):
    def __init__(self, dataset, tokenizer, return_ids=True, max_context_length=200, max_sequence_length=256, q_prefix=None, for_model=SWECTRL_MODEL, omit_token="[...]", start_codes=None, end_codes=None, distractor_limit=3, for_generation=False):
        super().__init__()
        self.__tok = tokenizer
        self.__for_generation = for_generation
        self.__return_ids = return_ids
        self.__conv_func = int if self.__return_ids else str
        self.__max_ctx_len = max_context_length
        self.__max_seq_len = max_sequence_length
        self.__q_prefix = q_prefix
        self.__omit_token = omit_token
        if self.__q_prefix:
            print("Training question prefix: {}".format(self.__q_prefix))
        else:
            print("No question prefix was set")
        # include allowance for [CLS] and [SEP] for BERT
        self.__for_model = for_model
        self.__scodes = start_codes
        self.__ecodes = end_codes
        self.__dis_limit = distractor_limit
        self.__encode(dataset)

    @property
    def tok(self):
        return self.__tok

    def __encode_qa(self, dp):
        enc_data = []

        ctx = dp["context"]

        func_name = 'encode' if self.__return_ids else 'tokenize'
        func = getattr(self.__tok, func_name)

        full_enc_ctx = func(ctx, add_special_tokens=False)
        N_ctx = len(full_enc_ctx)
        
        if self.__for_model == SWECTRL_MODEL:
            q_prefix = "{} ".format(self.__scodes["mcq"])
        else:
            q_prefix = ""

        if self.__for_generation:
            ctx_chunks = chunkify(full_enc_ctx, self.__max_ctx_len, return_last=True)
            
            if self.__q_prefix:
                q_prefix = "{}{}".format(q_prefix, self.__q_prefix)
            enc_q_prefix = func(q_prefix, add_special_tokens=False)
            return [
                {
                    'context': chunk,
                    "prefix": enc_q_prefix
                }
                for chunk in ctx_chunks
            ]
                
        q = dp["question"]
        a = dp["correct"]
        distractors = dp["distractors"]

        if not q.strip():
            return

        if self.__q_prefix:
            q_prefix = "{}{} {}".format(q_prefix, self.__q_prefix, q)
        else:
            q_prefix = "{}{}".format(q_prefix, q)
        enc_q_prefix = func(q_prefix, add_special_tokens=False)
        
        dis_prefixes = ["{})".format(x) for x in 'bcdefghijk']
        enc_dis = []
        for i, dis in enumerate(distractors[:self.__dis_limit]):
            enc_dis.append(
                func("{} {}".format(dis_prefixes[i], dis), add_special_tokens=False)
            )

        a_prefix = "a) {}".format(a)
        enc_a_prefix = func(a_prefix, add_special_tokens=False)

        if self.__for_model == SWECTRL_MODEL:
            s_control, e_control = None, None
            if "control" in dp and dp["control"]:
                s_control = func(self.__scodes[dp["control"]], add_special_tokens=False)
                e_control = func(self.__ecodes[dp["control"]], add_special_tokens=False)
        
        enc_text = enc_q_prefix + enc_a_prefix
        for enc_d in enc_dis:
            enc_text = enc_text + enc_d
        
        if self.__for_model == SWECTRL_MODEL:
            enc_text += func(self.__ecodes["mcq"], add_special_tokens=False)

        # make room for OCC and ECC in case of SweCTRL-Mini
        # or [CLS], and [SEP] between T and Q, Q and A, after A + [SEP] after each distractor
        delta = 2 if self.__for_model == SWECTRL_MODEL else (4 + len(enc_dis))
        Nt = len(enc_text)
        diff = self.__max_seq_len - self.__max_ctx_len - Nt - delta
        max_ctx_len = self.__max_ctx_len + min(0, diff)
            
        if N_ctx > max_ctx_len:
            # choose only part relevant for the correct answer surrounded by some text then
            key_basis = dp.get("correct_basis")
            enc_omit = func(self.__omit_token, add_special_tokens=False)
            if isinstance(key_basis, tuple) and len(key_basis) == 2:
                s, e = key_basis
                enc_ctx_basis = func(ctx[s:e], add_special_tokens=False)
                
                N_rem = max_ctx_len - len(enc_ctx_basis) - 2 * len(enc_omit) 
                if N_rem > 0:
                    prefix, suffix = ctx[:s], ctx[e:]
                    enc_prefix = func(prefix, add_special_tokens=False)
                    enc_suffix = func(suffix, add_special_tokens=False)
                   
                    N_prefix, N_suffix = len(enc_prefix), len(enc_suffix)

                    # TODO: balance the things
                    if N_prefix >= N_rem and N_suffix >= N_rem:
                        start_cutoff = random.randint(0, N_rem)
                        end_cutoff = N_rem - start_cutoff
                    elif N_suffix < N_rem:
                        end_cutoff = N_suffix
                        start_cutoff = N_rem - end_cutoff
                    else:
                        start_cutoff = N_prefix
                        end_cutoff = N_rem - start_cutoff

                    enc_ctx = enc_prefix[-start_cutoff:] + enc_ctx_basis + enc_suffix[:end_cutoff]
                else:
                    enc_ctx = func(
                        enc_ctx_basis,
                        add_special_tokens=False,
                        truncation=True,
                        max_length=max_ctx_len - 2 * len(enc_omit)
                    )
            else:
                enc_ctx = func(
                    ctx, add_special_tokens=False,
                    truncation=True,
                    max_length=max_ctx_len - 2 * len(enc_omit)
                )
            
            if enc_ctx[:3] != full_enc_ctx[:3]:
                # not a 100% guarantee that it will be a real match,
                # of the beginnings, since there can be the same words,
                # but it is a good enough proxy
                enc_ctx = enc_omit + enc_ctx

            if enc_ctx[-3:] != full_enc_ctx[-3:]:
                # similar heuristics for the endings as for the beginnings
                enc_ctx = enc_ctx + enc_omit
        else:
            enc_ctx = full_enc_ctx

        enc_data.append({
            "context": enc_ctx,
        })

        if self.__for_model == SWECTRL_MODEL:
            # For SweCTRL
            enc_data[-1].update({
                "mcq": enc_text,
                "s_control": s_control,
                "e_control": e_control
            })
        else:
            # For BERT
            enc_data[-1].update({
                "question": enc_q_prefix,
                "correct": enc_a_prefix,
                "distractors": enc_dis
            })

        return enc_data

    def _encode_chunk(self, chunk, position=0):
        enc_data = []
        for dp in chunk:#, position=position):
            for eqa in self.__encode_qa(dp):
                enc_data.append(eqa)
        return enc_data

    def _encode_chunk_wrapper(self, args):
        return self._encode_chunk(*args)

    def __encode(self, dataset):
        num_workers = mp.cpu_count() - 2

        N = len(dataset)
        k, m = divmod(N, num_workers)

        with mp.Pool(num_workers) as p:
            data = p.map(
                self._encode_chunk_wrapper,
                [(dataset[i*k+min(i, m):(i+1)*k+min(i+1, m)],  num_workers + (i+1)) for i in range(num_workers)]
            )
            for x in data:
                self._data.extend(x)


class BERTAregLeftToRightTransform(GenericDataset):
    def __init__(self, dataset, for_generation=False, permute_distractors=False):
        super().__init__()
        self.__for_generation = for_generation
        self.__permute_distractors = permute_distractors

        self.__encode(dataset)
        
    def __encode(self, dataset):
        def add_datapoints(seq):
            for idx in range(len(seq)):
                if idx > 0:
                    input_ids.append(seq[idx-1])
                    tt_ids.append(1)
                    att_mask.append(1)

                masked_input_ids = list(input_ids)
                masked_input_ids.append(tok.mask_token_id)
                labels = [IGNORE_INDEX] * len(masked_input_ids)
                labels[-1] = seq[idx]
                masked_tt_ids = list(tt_ids)
                masked_tt_ids.append(1)
                masked_att = list(att_mask)
                masked_att.append(1)
                
                self._data.append({
                    "input_ids": masked_input_ids,
                    "token_type_ids": masked_tt_ids,
                    "attention_mask": masked_att,
                    "labels": labels
                })
            else:
                input_ids.append(seq[idx])
                tt_ids.append(1)
                att_mask.append(1)

                masked_input_ids = list(input_ids)
                masked_input_ids.append(tok.mask_token_id)
                labels = [IGNORE_INDEX] * len(masked_input_ids)
                labels[-1] = tok.sep_token_id
                masked_tt_ids = list(tt_ids)
                masked_tt_ids.append(1)
                masked_att = list(att_mask)
                masked_att.append(1)

                self._data.append({
                    "input_ids": masked_input_ids,
                    "token_type_ids": masked_tt_ids,
                    "attention_mask": masked_att,
                    "labels": labels
                })

                input_ids.append(tok.sep_token_id)
                tt_ids.append(1)
                att_mask.append(1)

        assert hasattr(dataset, "tok"), "Run TokenizeTransform first"
        tok = dataset.tok
        for dp in dataset:
            inp = tok.prepare_for_model(dp["context"])
            input_ids = list(inp["input_ids"])
            tt_ids = list(inp["token_type_ids"])
            att_mask = list(inp["attention_mask"])

            if self.__for_generation:
                N_prefix = len(dp['prefix'])
                input_ids.extend(dp['prefix'])
                tt_ids.extend([1] * N_prefix)
                att_mask.extend([1] * N_prefix)

                input_ids.append(tok.mask_token_id)
                tt_ids.append(1)
                att_mask.append(1)
                
                self._data.append({
                    "input_ids": input_ids,
                    "token_type_ids": tt_ids,
                    "attention_mask": att_mask
                })
            else:
                add_datapoints(dp["question"])
                add_datapoints(dp["correct"])
            
                if self.__permute_distractors:
                    for dset in permutations(dp["distractors"]):
                        old_input_ids = copy.deepcopy(input_ids)
                        old_tt_ids = copy.deepcopy(tt_ids)
                        old_att_mask = copy.deepcopy(att_mask)
                        for d_tok in dset:
                            type_id = add_datapoints(d_tok)
                        input_ids = old_input_ids
                        tt_ids = old_tt_ids
                        att_mask = old_att_mask
                else:
                    for d_tok in dp["distractors"]:
                        type_id = add_datapoints(d_tok)


class BERTAregArbitraryOrderTransform(GenericDataset):
    def __init__(self, dataset, separate_distractor_type=False, for_generation=False, permute_distractors=False,
        number_of_samples=20, sampling_strategy='cap', max_stem_length=30, max_alt_length=20, seed=42, all_at_once=False):
        super().__init__()
        self.__for_generation = for_generation
        self.__permute_distractors = permute_distractors
        self.__num_samples = number_of_samples
        self.__seed = seed
        self.__sampling_strategy = sampling_strategy
        self.__max_stem = max_stem_length
        self.__max_alt = max_alt_length
        self.__all_at_once = all_at_once

        self.__blank_token = dataset.tok.additional_special_tokens[-1]
        self.__blank_token_id = dataset.tok.convert_tokens_to_ids(self.__blank_token)

        self.__encode(dataset)

    def __collect_q_distribution(self, dataset):
        for dp in dataset:
            for i, token in enumerate(dp["question"]):
                self.__q_pos_cnt[i].append(token)

            if self.__max_qlength > 0:
                for j in range(i, self.__max_qlength):
                    self.__q_pos_cnt[j].append(self.__blank_token_id)

        for i in self.__q_pos_cnt:
            self.__q_pos_cnt[i] = Counter(self.__q_pos_cnt[i])

    def __encode(self, dataset):
        def add_datapoints(seq, prev_seq):
            N = len(seq)

            if self.__sampling_strategy == 'cap':
                n_samples = min(N, self.__num_samples)
            elif self.__sampling_strategy.startswith('cap_'):
                try:
                    cap_ratio = float(self.__sampling_strategy.split("_")[1])
                except:
                    print("WARNING: failed to parse cap_ratio, taking 1.0")
                    cap_ratio = 1.0
                n_samples = min(int(cap_ratio * N), self.__num_samples)
            else:
                n_samples = self.__num_samples

            for _ in range(n_samples):
                masked_ratio = random.random()

                masked_input_ids = list(input_ids)
                masked_tt_ids = list(tt_ids)
                masked_att = list(att_mask)
                labels = [IGNORE_INDEX] * len(masked_input_ids)

                for pseq in prev_seq:
                    for p_tok in pseq:
                        masked_input_ids.append(p_tok)
                        masked_tt_ids.append(1)
                        masked_att.append(1)
                        labels.append(IGNORE_INDEX)
                    masked_input_ids.append(tok.sep_token_id)
                    masked_tt_ids.append(1)
                    masked_att.append(1)
                    labels.append(IGNORE_INDEX)

                indeed_masked, masked_pos = 0, []
                for idx in range(N):
                    is_masked = (N == 1) or random.random() >= masked_ratio
                    
                    if is_masked and seq[idx] != tok.sep_token_id:
                        indeed_masked += 1
                        masked_input_ids.append(tok.mask_token_id)
                        labels.append(seq[idx])
                    else:
                        masked_input_ids.append(seq[idx])
                        labels.append(IGNORE_INDEX)
                    
                    masked_tt_ids.append(1)
                    masked_att.append(1)

                if indeed_masked == 0: # at least one masked token except [SEP]
                    idx = random.randint(0, N)
                    masked_input_ids[-idx] = tok.mask_token_id
                    labels[-idx] = seq[-idx]

                N_inp = len(masked_input_ids)

                self._data.append({
                    "input_ids": masked_input_ids,
                    "token_type_ids": masked_tt_ids,
                    "attention_mask": masked_att,
                    "labels": labels,
                })

        def pad_with_blank(seq, max_len):
            pad_len = max_len - len(seq)
            if pad_len < 0:
                print("WARNING: Maximum length is smaller than an item in the dataset, skipping")
                return False
            for _ in range(pad_len):
                seq.append(self.__blank_token_id) # add [BLANK] tokens
            return True

        assert hasattr(dataset, "tok"), "Run TokenizeTransform first" 
        tok = dataset.tok
        
        if self.__seed:
            random.seed(self.__seed) # for repeatability
            np.random.seed(self.__seed) # TODO: potentially change to BitGenerator?
        
        for dp in dataset:
            inp = tok.prepare_for_model(dp["context"])
            input_ids = list(inp["input_ids"])
            tt_ids = list(inp["token_type_ids"])
            att_mask = list(inp["attention_mask"])
            

            if self.__for_generation:
                self._data.append({
                    "input_ids": input_ids,
                    "token_type_ids": tt_ids,
                    "attention_mask": att_mask,
                })
            else:
                if self.__max_stem > 0:
                    # Make sure that max question length is larger or equal to the longest question in the dataset
                    is_padded = pad_with_blank(dp["question"], self.__max_stem)
                    if not is_padded:
                        continue
                
                if self.__max_alt > 0:
                    # Make sure that max question length is larger or equal to the longest question in the dataset
                    is_padded = pad_with_blank(dp["correct"], self.__max_alt)
                    if not is_padded:
                        continue

                    all_dis_padded = True
                    for dis_id in range(len(dp["distractors"])):
                        is_padded = pad_with_blank(dp["distractors"][dis_id], self.__max_alt)
                        all_dis_padded = all_dis_padded and is_padded
                    if not all_dis_padded:
                        continue

                if self.__all_at_once:
                    to_add = dp["question"] + [tok.sep_token_id] + dp["correct"] 
                    
                    if self.__permute_distractors:
                        for dset in permutations(dp["distractors"]):
                            mcq = to_add
                            for d_tok in dset:
                                mcq += [tok.sep_token_id] + d_tok
                            add_datapoints(mcq, [])
                    else:
                        mcq = to_add
                        for d_tok in dp["distractors"]:
                            mcq += [tok.sep_token_id] + d_tok
                        add_datapoints(mcq, [])
                else:
                    add_datapoints(dp["question"], [])
                    add_datapoints(dp["correct"], [dp["question"]])
                    
                    prev_items = [dp["question"], dp["correct"]]
                    if self.__permute_distractors:
                        for dset in permutations(dp["distractors"]):
                            old_input_ids = copy.deepcopy(input_ids)
                            old_tt_ids = copy.deepcopy(tt_ids)
                            old_att_mask = copy.deepcopy(att_mask)
                            for d_idx, d_tok in enumerate(dset):
                                prev = prev_items + dset[:d_idx]
                                add_datapoints(d_tok, prev)
                                
                            input_ids = old_input_ids
                            tt_ids = old_tt_ids
                            att_mask = old_att_mask
                    else:
                        for d_idx, d_tok in enumerate(dp["distractors"]):
                            prev = prev_items + dp["distractors"][:d_idx]
                            add_datapoints(d_tok, prev)


class CTRLAregLeftToRightTransform(GenericDataset):
    def __init__(self, dataset=None, max_sequence_length=256, for_generation=False):
        super().__init__()

        self.__max_seq_len = max_sequence_length
        self.__for_generation = for_generation

        if dataset:
            self.__encode(dataset)
        
    def __encode(self, dataset):
        assert hasattr(dataset, "tok"), "Run TokenizeTransform first"
        tok = dataset.tok

        for dp in tqdm(dataset):
            prepend = dp.get("context", [])
            if self.__for_generation:
                prefix = dp.get("prefix", [])
                self._data.append(
                    tok.prepare_for_model(prepend + prefix, return_token_type_ids=False)
                )
                continue

            s_control_code = dp["s_control"] if dp.get("s_control", False) else []
            e_control_code = dp["e_control"] if dp.get("e_control", False) else []

            # s_control_code = text control code
            # prepend = text + 
            final_text = s_control_code + prepend + dp["mcq"] + e_control_code
            text_len = len(final_text)
            for i in range(0, text_len, self.__max_seq_len):
                inp = tok.prepare_for_model(
                    final_text[i:i+self.__max_seq_len],
                    return_token_type_ids=False
                )
                # From https://huggingface.co/docs/transformers/model_doc/ctrl
                # Note that the labels are shifted inside the model, i.e. you can set labels = input_ids 
                # Indices are selected in [-100, 0, ..., config.vocab_size]
                # All labels set to -100 are ignored (masked), the loss is only computed for labels in [0, ..., config.vocab_size]
                # v This is why we don't shift the labels here!
                inp["labels"] = list(inp["input_ids"])
                # Ignore the things related to text, since we want to predict only the stem, key and distractors!
                for j in range(len(prepend) + (len(s_control_code) > 0)):
                    inp["labels"][j] = -100
                self._data.append(inp)

    def _to_list(self):
        return self._data

    def _from_list(self, lst):
        self._data = lst


if __name__ == '__main__':
    from transformers import AutoTokenizer
   
    test_mode = SWECTRL_MODEL
    formulation = AREG_LTR
    data_mode = SWEQUAD_DS
    
    if data_mode == SWEQUAD_DS:
        ds = SweQuadMCDataset(["data/swequad-mc/training.json"])
    elif data_mode == QUASI_DS:
        ds = QuasiDataset(["data/quasi/quasi.json"])
    else:
        sys.exit(1)
    
    # leave 64 tokens for question and 3 distractors
    if test_mode == KBBERT_MODEL:
        tok = AutoTokenizer.from_pretrained(KBBERT_HF_KEY)
        tok.add_special_tokens({
            'additional_special_tokens': tok.additional_special_tokens + [BLANK_TOKEN]
        })
        ds = TokenizeTransform(
            ds, tok, return_ids=False,
            q_prefix="Fråga:",
            for_model=KBBERT_MODEL,
            max_context_length=384,
            #max_context_length=441,
            max_sequence_length=512,
            for_generation=True
        )
        if formulation == AREG_LTR:
            ds = BERTAregLeftToRightTransform(ds, for_generation=True)
        elif formulation == AREG_ARB:
            ds = BERTAregArbitraryOrderTransform(ds, all_at_once=True,
                    for_generation=True)
    elif test_mode == SWECTRL_MODEL:
        tok = AutoTokenizer.from_pretrained(SWECTRL_HF_KEY)
        tok, START_C_CODES, END_C_CODES = add_task_special_tokens(tok)

        ds = TokenizeTransform(
            ds, tok, return_ids=True,
            q_prefix="Fråga:",
            for_model=SWECTRL_MODEL,
            max_context_length=192,
            max_sequence_length=256,
            start_codes=START_C_CODES,
            end_codes=END_C_CODES,
            for_generation=True
        )
        ds = CTRLAregLeftToRightTransform(ds, for_generation=True)

    for i, x in enumerate(ds):
        print(x)

        if i == 30:
            break
    print(tok.encode(START_C_CODES['mcq']))
