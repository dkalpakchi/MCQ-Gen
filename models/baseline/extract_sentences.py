import json
import argparse

import jsonlines as jsl

import stanza


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--lang', type=str, help='A language for template generation (en, sv are currently supported)')
    parser.add_argument('-d', '--data', type=str, help='Comma-separated list of files to generate questions from')
    parser.add_argument('-o', '--output-file', type=str, default="result.jsonl")
    args = parser.parse_args()
    
    sv = stanza.Pipeline(lang=args.lang, processors='tokenize')

    l2i = {v: i for i, v in enumerate('abcd')}
    extracted_data = []
    for fname in sorted(args.data.split(",")):
        with open(fname) as f:
            data = json.load(f)

            for x in data:
                doc = sv(x['text'])
                for item in x["test"]:
                    choices = item["mcq"]["choices"]
                    # remove the letters (a - d)
                    choices = [x[2:].strip() for x in choices]

                    key = item['annotations']["correct"]
                    key_basis = item['annotations']['bases'][key]

                    if not key_basis: continue

                    found_sentence = False
                    for sentence in doc.sentences:
                        s = x['text'].index(sentence.text)
                        e = s + len(sentence.text)
                        #print(sentence.text, key_basis, s, e)
                        if key_basis['start'] >= s and key_basis['end'] <= e:
                            if key_basis['text'] in sentence.text:
                                found_sentence = sentence.text
                                break

                    if found_sentence:
                        extracted_data.append({
                            "sentence": found_sentence,
                            "question": item["mcq"]["stem"].partition(" ")[2],
                            "answer": choices[l2i[key]]
                        })
                    else:
                        print(key_basis)
                        print("SENTENCE NOT FOUND!!!\n")

    with jsl.open(args.output_file, 'w') as writer:
        for obj in extracted_data:
            writer.write(obj)
