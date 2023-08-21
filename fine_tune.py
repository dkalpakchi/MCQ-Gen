import os
import sys
import json
import logging

from tqdm import tqdm

import torch

import transformers
from transformers import (
    AutoTokenizer, AutoModel,
    BertConfig, BertForMaskedLM,
    CTRLConfig, CTRLModel, CTRLLMHeadModel,
    HfArgumentParser, TrainingArguments, Trainer,
    DataCollatorForTokenClassification, PreTrainedTokenizerFast
)

from transformers.trainer_utils import get_last_checkpoint

from common import (
    SWECTRL_MODEL, KBBERT_MODEL,
    SWECTRL_HF_KEY, KBBERT_HF_KEY,
    SWEQUAD_DS, QUASI_DS,
    AREG_LTR, AREG_ARB, BLANK_TOKEN
)
from dataset import (
    TokenizeTransform, SweQuadMCDataset, QuasiDataset,
    BERTAregLeftToRightTransform, CTRLAregLeftToRightTransform,
    BERTAregArbitraryOrderTransform
)
from util import FineTuningArguments, add_task_special_tokens


logger = logging.getLogger(__name__)

class FilePrinterCallback(transformers.TrainerCallback):
    def __init__(self, fname):
        self.__file = open(fname, 'a+')

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero:
            self.__file.write("{}\n".format(json.dumps(logs)))

    def on_train_end(self, args, state, control, **kwargs):
        """
        Event called at the end of training.
        """
        self.__file.close()


if __name__ == '__main__':
    parser = HfArgumentParser((TrainingArguments, FineTuningArguments))
    train_args, ft_args = parser.parse_args_into_dataclasses()
    logger.info(train_args)
    logger.info(ft_args)
    logger.info("Torch version: {}".format(torch.__version__))
    logger.info("Transformers version: {}".format(transformers.__version__))
 
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    
    transformers.logging.set_verbosity_info()

    if ft_args.loader == SWEQUAD_DS:
        ds = SweQuadMCDataset(ft_args.train_data.split(","))
    elif ft_args.loader == QUASI_DS:
        ds = QuasiDataset(ft_args.train_data.split(","))

    if ft_args.base_model == KBBERT_MODEL:
        tokenizer = AutoTokenizer.from_pretrained(KBBERT_HF_KEY)
        max_ctx_len = 441 # ~64 * 1.1 tokens = ~71 tokens for MCQ
        max_seq_len = 512
        
        if ft_args.formulation == AREG_ARB:
            max_ctx_len = 384 # to accommodate fixed lengths of stems and alternatives
            tokenizer.add_special_tokens({
                'additional_special_tokens': tokenizer.additional_special_tokens + [BLANK_TOKEN]
            })
        
        ds = TokenizeTransform(
            ds, tokenizer, return_ids=True,
            q_prefix=ft_args.q_prefix,
            for_model=ft_args.base_model,
            max_context_length=max_ctx_len,
            max_sequence_length=max_seq_len
        )
        
        if ft_args.formulation == AREG_LTR:
            train_ds = BERTAregLeftToRightTransform(ds)
        elif ft_args.formulation == AREG_ARB:
            train_ds = BERTAregArbitraryOrderTransform(
                ds, all_at_once=ft_args.all_at_once
            )
        
        model = BertForMaskedLM.from_pretrained(KBBERT_HF_KEY)
        
        if ft_args.formulation == AREG_ARB:
            model.resize_token_embeddings(len(tokenizer))
    elif ft_args.base_model == SWECTRL_MODEL:
        tokenizer = AutoTokenizer.from_pretrained(SWECTRL_HF_KEY)
        tokenizer, START_C_CODES, END_C_CODES = add_task_special_tokens(tokenizer)

        max_ctx_len = 192 # 64 tokens for MCQ
        max_seq_len = 256
        ds = TokenizeTransform(
            ds, tokenizer, return_ids=True,
            q_prefix=ft_args.q_prefix,
            for_model=ft_args.base_model,
            max_context_length=max_ctx_len,
            max_sequence_length=max_seq_len,
            start_codes=START_C_CODES,
            end_codes=END_C_CODES
        )
        train_ds = CTRLAregLeftToRightTransform(ds)

        model = CTRLLMHeadModel.from_pretrained(SWECTRL_HF_KEY)
        model.resize_token_embeddings(len(tokenizer))
    
    logger.info(model)

    collator = DataCollatorForTokenClassification(tokenizer)
    
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(train_args.output_dir) and train_args.do_train and not train_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(train_args.output_dir)
        if last_checkpoint is None and len(os.listdir(train_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({train_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and train_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
   
    checkpoint = None
    if train_args.resume_from_checkpoint is not None:
        checkpoint = train_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    logger.debug(len(train_ds))
    logger.info("Number of params: {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad))) 

    if not os.path.exists(train_args.output_dir):
        os.makedirs(train_args.output_dir)
    
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        data_collator=collator,
        callbacks=[FilePrinterCallback(os.path.join(train_args.output_dir, "logs.jsonl"))]
    )

    if train_args.do_train:
        ft_args.max_context_length = max_ctx_len
        ft_args.max_sequence_length = max_seq_len
        torch.save(ft_args, os.path.join(train_args.output_dir, 'ft_args.bin'))
        
        with open(os.path.join(train_args.output_dir, "versions.json"), 'w') as f:
            json.dump({
                "torch_version": torch.__version__,
                "transformers_version": transformers.__version__
            }, f)
     
        trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
