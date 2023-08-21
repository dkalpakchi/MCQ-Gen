import argparse
import math

import jsonlines as jsl
from tqdm import tqdm

import stanza
import quinductor as qi
import udon2
from udon2.kernels import ConvPartialTreeKernel

from dataset import SweQuadMCDataset, TextualDatasetFromYAML


def generate_distractors(key_root, kernel, corpus):
    key_N = key_root.subtree_size() + 1
    key_k = math.sqrt(kernel(key_root, key_root))

    ranks = []
    calculated = set()

    for doc_nodes in corpus: 
        for n in doc_nodes:
            survivors = n.select_by('upos', key_root.upos)
            for s in survivors:
                s_str = s.get_subtree_text()
                if s_str in calculated: continue

                if s.has_all("feats", str(key_root.feats)) and not s.parent.is_root():
                    Ns = s.subtree_size() + 1
                    if (key_N > 1 and Ns <= 1) or (key_N <= 1 and Ns > 1): continue
                    ss_k = math.sqrt(kernel(s, s))
                    ranks.append((s, kernel(key_root, s) / (key_k * ss_k)))
                    calculated.add(s_str)
    ranks = sorted(ranks, key=lambda x: x[1], reverse=True)
    return [r[0].get_subtree_text() for r in ranks[:3]]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--lang', default='sv', type=str, help='A language for template generation (en, sv are currently supported)')
    parser.add_argument('-d', '--dataset', type=str, help="Dataset path")
    parser.add_argument('-tr', '--train', type=str, help="A file with the text corpus for distractors")
    parser.add_argument('-tf', '--templates-folder', type=str, help="A folder with Quinductor templates")
    parser.add_argument('-o', '--output', type=str, help="Suffix for the output file")
    parser.add_argument('-n', '--nsamples', default=1, type=int)
    args = parser.parse_args()

    qi_tools = qi.use(args.lang, templates_folder=args.templates_folder)

    dep_proc = 'tokenize,lemma,pos,depparse'
    sv = stanza.Pipeline(lang=args.lang, processors=dep_proc)

    problems = 0
    fname, afname = 'text.conll', 'answer.conll'
    kernel = ConvPartialTreeKernel(
        'GRCT', includeForm=False, includeFeats=True
    )

    train_ds = SweQuadMCDataset(args.train.split(","))
    corpus = [x["context"] for x in train_ds]
    
    parsed_corpus = []
    for raw_text in tqdm(corpus):
        parsed_corpus.append(udon2.Importer.from_stanza(sv(raw_text).to_dict()))

    # Dataset
    ds = TextualDatasetFromYAML(args.dataset.split(","))
    
    choices_letters = list("abcd")

    num_mcqs = args.nsamples
    if args.output:
        out_fname = "ud_baseline_{}.jsonl".format(args.output)
    else:
        out_fname = "ud_baseline.jsonl"
    with jsl.open(out_fname, 'w') as writer:
        for ds_dp in ds:
            text = ds_dp["context"]
            text_doc = sv(text)
            text_nodes = udon2.Importer.from_stanza(text_doc.to_dict())
            res = qi.generate_questions(text_nodes, qi_tools)

            dp = {
                "text": text,
                "mcqs": []
            }

            text_root = text_nodes[0]
            for pair in res[:num_mcqs]:
                q, a = pair.q, pair.a
                a_nodes = text_root.select_by("form", pair.a)
                if len(a_nodes) == 0:
                    continue
                dis = generate_distractors(a_nodes[0], kernel, parsed_corpus)
                choices = [a]
                choices.extend(dis)
                dp["mcqs"].append({
                    "stem": q,
                    "choices": ["{}) {}".format(l, c) for l, c in zip(choices_letters, choices)]
                })
            writer.write(dp)
