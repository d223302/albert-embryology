#!/usr/bin/env python
import json
import numpy as np
import glob
import os
import argparse
from transformers import BertTokenizer, AlbertTokenizer, ElectraTokenizer

def convert(rel, data):
  obj = data['obj_label']
  sub = data['sub_label']
  gt = rel['template'].replace('[X]', sub)
  a, b = gt.split('[Y]')
  a = a.rstrip().lstrip()
  b = b.rstrip().lstrip()
  a = tokenizer.encode(a, add_special_tokens = False)
  a.insert(0, tokenizer.cls_token_id)
  if b != '.':
    b = tokenizer.encode(b, add_special_tokens = False)
  else:
    b = tokenizer.encode('.', add_special_tokens = False)[-1:]
  b.append(tokenizer.sep_token_id)
  span_start = len(a)
  Y = tokenizer.encode(obj.rstrip().lstrip(), add_special_tokens = False)
  candidate = Y
  span_end = len(a) + len(Y)
  unmasked = a + Y + b
  masked = unmasked.copy()
  for i in range(span_start, span_end):
    masked[i] = tokenizer.mask_token_id
  return masked, unmasked, span_start, span_end, candidate

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("-m", "--model", type = str, help = "LM used to extract features")
  parser.add_argument("-i", "--input", type = str, help = "Directory where you place the unziped LAMA dataset")
  parser.add_argument("-t", "--tokenizer", type = str, help = "path to vocab file for tokenizer")
  args = parser.parse_args()


  if 'albert' in args.model:
    model_type = 'albert'
    tokenizer = AlbertTokenizer(vocab_file = args.tokenizer)
  elif 'bert' in args.model:
    model_type = 'bert'
    tokenizer = BertTokenizer(vocab_file = args.tokenizer)
  else:
    raise NotImplementedError("The model is currently not supported")

  relation = {}
  with open(os.path.join(args.input, 'LAMA/relations.jsonl'), 'r') as f:
    lines = list(f)
  for line in lines:
    data = json.loads(line)
    if data['type'] == 'N-M': continue
    relation[data['relation']] = {'template': data['template'], 'rel_type': data['type']}

  json_files = glob.glob(os.path.join(args.input, 'LAMA/TREx', '**/*.jsonl'), recursive = True)
  for jsonl in json_files:
    label = []
    rel_name = jsonl.split('/')[-1].split('.')[0]
    if rel_name not in relation:
      continue
    train_data = {}
    with open(jsonl, 'r') as f:
      lines = list(f)

    for line in lines:
      data = json.loads(line)
      masked, unmasked, span_start, span_end, cand  = convert(relation[rel_name], data)
      if span_end - span_start > 1: continue # Here we only consider the case when target length is 1
      label.append(cand)
      train_data[data['uuid']] = {'masked': masked, 'unmasked': unmasked, 'start': span_start, 'end': span_end}
    output_path = os.path.join(args.input, 'LAMA/TREx_processed')
    if not os.path.exists(output_path):
      os.makedirs(output_path)
    with open(os.path.join(output_path, rel_name + '.json'), 'w') as f:
      json.dump(train_data, f)
