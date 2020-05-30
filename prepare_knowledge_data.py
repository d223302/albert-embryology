#!/usr/bin/env python
import json
import numpy as np
import glob
import os
from transformers import AlbertTokenizer
tokenizer = AlbertTokenizer(vocab_file = '/work/dcml0714/albert/albert_base/30k-clean.model')

def convert(rel, data):
  obj = data['obj_label']
  sub = data['sub_label']
  gt = rel['template'].replace('[X]', sub)
  a, b = gt.split('[Y]')
  a = a.rstrip().lstrip()
  b = b.rstrip().lstrip()
  a = tokenizer.encode(a, add_special_tokens = False)
  a = [2] + a
  if b != '.':
    b = tokenizer.encode(b, add_special_tokens = False)
  else:
    b = [9]
  b = b + [3]
  span_start = len(a)
  Y = tokenizer.encode(obj, add_special_tokens = False)
  candidate = Y[0]
  span_end = len(a) + len(Y)
  unmasked = a + Y + b
  masked = unmasked.copy()
  for i in range(span_start, span_end):
    masked[i] = 4
  return masked, unmasked, span_start, span_end, candidate

relation = {}
with open('/work/dcml0714/LAMA/relations.jsonl', 'r') as f:
  lines = list(f)
  
for line in lines: 
  data = json.loads(line) 
  if data['type'] == 'N-M': continue
  relation[data['relation']] = {'template': data['template'], 'rel_type': data['type']}

json_files = glob.glob('/work/dcml0714/LAMA/TREx' + '**/*.jsonl', recursive = True)

for jsonl in json_files:
  false_cand = []  
  rel_name = jsonl.split('/')[-1].split('.')[0]
  print(rel_name)
  if rel_name not in relation:
    continue  
  train_data = {}
  with open(jsonl, 'r') as f:
    lines = list(f)
  
  for line in lines:
    data = json.loads(line)
    masked, unmasked, span_start, span_end, cand  = convert(relation[rel_name], data)
    if span_end - span_start > 1: continue
    false_cand.append(cand)
    train_data[data['uuid']] = {'masked': masked, 'unmasked': unmasked, 'start': span_start, 'end': span_end}
  output_path = os.path.join('/work/dcml0714/LAMA/TREx_processed', rel_name + '.json')  
  with open(output_path, 'w') as f:    
    json.dump(train_data, f)  
  np.save(os.path.join('/work/dcml0714/LAMA/TREx_processed/fake', rel_name + '.npz'), false_cand)  

