#!/usr/bin/env python3
import glob
import torch
from transformers import AlbertTokenizer, AlbertModel, AlbertForMaskedLM, AlbertConfig
import numpy as np
import json
from tqdm import tqdm
import sys


tokenizer = AlbertTokenizer(vocab_file = '/work/dcml0714/albert/albert_base/30k-clean.model')
config = AlbertConfig.from_json_file('/work/dcml0714/albert/albert_base/albert_config.json')

config.output_hidden_states = True
config.output_attentions = True
model = AlbertForMaskedLM.from_pretrained(pretrained_model_name_or_path = None,
          config = config,
          state_dict = torch.load('/work/dcml0714/albert/pytorch_model/pytorch_model_' + sys.argv[1] + '.bin'))
model.eval().cuda()

rel_names = glob.glob('/work/dcml0714/LAMA/TREx_processed/' + '**/*.json', recursive = True)

result = {}
result_false = {}

for rel_name in rel_names:
  result[rel_name.split('/')[-1].split('.')[0]] = [0, 0]

for rel in tqdm(rel_names):
  f = open(rel, 'r')
  f_dict = json.load(f)
  f.close()

#"""
  for k, data in f_dict.items():
    tokens_tensor = torch.tensor([data['masked']])
    
    
    with torch.no_grad():
      _, prediction_scores = model(tokens_tensor.cuda(),masked_lm_labels = tokens_tensor.cuda())[:2]
    start = int(data['start'])
    gt = data['unmasked'][start]
   

    prediction_scores = prediction_scores.squeeze().argmax(-1).cpu().numpy()[start]
    result[rel.split('/')[-1].split('.')[0]][1] += 1
    if gt == prediction_scores:
      result[rel.split('/')[-1].split('.')[0]][0] += 1

accuracy = []

for rel in result:
  accuracy.append(str(result[rel][0] / (result[rel][1] + 1e-10)))
accuracy = ','.join(accuracy)
accuracy = sys.argv[1] + ',' + accuracy + '\n'
with open('/home/dcml0714/albert_embryo/exp_result/pretrain/knowledge_predict.txt', 'a+') as f:
  f.write(accuracy)  

