#!/usr/bin/env python3
import numpy as np
import sys
import argparse
from transformers import BertTokenizer, BertForMaskedLM, BertConfig,\
                         AlbertTokenizer, AlbertForMaskedLM, AlbertConfig
import os                         
import torch
import json
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--type_of_model", default = 'albert', help = "pretrained LM type")
parser.add_argument("-d", "--data", help = "path where you put your processed ontonotes data")
parser.add_argument("-o", "--output", help = "output file")
parser.add_argument("--config_and_vocab", help = "path to config.json and vocab.model")
args = parser.parse_args()
if args.type_of_model == 'albert':
  tokenizer = AlbertTokenizer(os.path.join(args.config_and_vocab, args.type_of_model, 'vocab.model'))
elif args.type_of_model == 'bert':
  tokenizer = BertTokenizer(os.path.join(args.config_and_vocab, args.type_of_model, 'vocab.model'))
else:
  raise NotImplementedError("The given model type %s is not supported" % args.type_of_model)


tag = {}
with open(os.path.join(args.data, 'ontonotes/const/pos/labels.txt')) as f:
  while True:
    pos = f.readline().rstrip()
    if pos == "": break
    tag[pos] = [] 
    
text_file = open(os.path.join(args.data, 'ontonotes/const/pos/conll-2012-test.json'), 'r')
for i, line in tqdm(enumerate(text_file.readlines())):
  data = json.loads(line)
  tokens = data['text'].split(' ')
  labels = data['targets']
  re_2_o = []
  retokenized = []
  for word_id, token in enumerate(tokens):
    token = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token))
    retokenized.extend(token)
    re_2_o.extend([word_id for _ in range(len(token))])
  retokenized = np.asarray(retokenized)  
  for span in labels:
    span1 = []
    for pos in range(span['span1'][0], span['span1'][1]):
      select = np.where(np.asarray(re_2_o) == pos)[0]
      span1.extend(select)
    predict = retokenized[np.asarray(span1)]
    tag[span['label']].extend(predict)

result = []
for key in tag:
  result.append(str(np.mean(tag[key])))
result = ','.join(result)


with open(os.path.join(args.output, args.type_of_model + "_count.txt"), 'a+') as f:
  f.write(result + '\n')  
