import math
import dataclasses as dc

import numpy as np
from numpy.random import default_rng
from numpy.linalg import lstsq
from scipy.linalg import orth

import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerFast

from common import UNKNOWN_TOKEN, PAD_TOKEN
from control_codes import START_C_CODES, END_C_CODES, ADD_START_C_CODES, ADD_END_C_CODES


@dc.dataclass
class FineTuningArguments:
    train_data: str = dc.field(
        default="data/swequad-mc/training.json",
        metadata={"help": "A CSV list of training data files"}
    )

    q_prefix: str = dc.field(
        default="Fråga:",
        metadata={"help": "prefix for questions"}
    )
    
    loader: str = dc.field(
        default="swequad-mc",
        metadata={"help": "swequad-mc or quasi"}
    )

    base_model: str = dc.field(
        default="swectrl-mini",
        metadata={"help": "Base model for fine-tuning: kb/bert or swectrl-mini"}
    )

    formulation: str = dc.field(
        default="areg_ltr",
        metadata={"help": "Mode for KB/BERT, one of areg_ltr and areg_arb"}
    )
    
    all_at_once: bool = dc.field(
        default=False,
        metadata={"help": "Whether to construct datapoints from the whole MCQ at once (words only if `train_mode` is `areg_arb`)"}
    )


@dc.dataclass
class LoraTuningArguments:
    lora_r: int = dc.field(
        default=4,
        metadata={"help": "r for LoRA"}
    )


class GradientPrinter:
    def __init__(self, name):
        self.name = name

    def __call__(self, grad):
        np_grad = grad.cpu().numpy()
        print("======== GRAD FOR {} ========".format(self.name))
        print("\tGRAD {}".format(grad))
        print("\tGRAD NORM {}".format(np.linalg.norm(np_grad)))
        print("\tGRAD MEAN {}".format(np.mean(np_grad)))
        print()


def add_task_special_tokens(tok):
    tok.add_special_tokens({'additional_special_tokens': list(ADD_START_C_CODES.values()) + list(ADD_END_C_CODES.values()) })

    START_C_CODES.update(ADD_START_C_CODES)
    END_C_CODES.update(ADD_END_C_CODES)
    return tok, START_C_CODES, END_C_CODES


def reassign_embeddings(model, new_emb):
    # Build new embeddings
    new_embeddings = nn.Embedding(*new_emb.shape).to(model.base_model.device)
    new_embeddings.weight.data = torch.tensor(new_emb).float()
    model.transformer.set_input_embeddings(new_embeddings)


def find_orth(O):
    # Taken from: https://stackoverflow.com/questions/50660389/generate-a-vector-that-is-orthogonal-to-a-set-of-other-vectors-in-any-dimension
    rand_vec = np.random.rand(O.shape[0], 1)
    A = np.hstack((O, rand_vec))
    b = np.zeros(O.shape[1] + 1)
    b[-1] = 1
    return lstsq(A.T, b)[0]


def neg_gauss_weight(x, mu, sigma):
    return math.exp(-(x - mu) ** 2 / (2 * sigma**2)) / (sigma * math.sqrt(2 * math.pi))


def random_choice_prob_index(a, axis=1):
    # Based on https://stackoverflow.com/questions/47722005/vectorizing-numpy-random-choice-for-given-2d-array-of-probabilities-along-an-a
    r = np.expand_dims(np.random.rand(a.shape[1-axis]), axis=axis)
    return (a.cumsum(axis=axis) > r).argmax(axis=axis)


def random_choice_logits_index(a, axis=1):
    rng = default_rng()
    return (a + rng.gumbel(size=a.shape)).argmax(axis=axis)
