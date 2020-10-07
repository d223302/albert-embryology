#!/usr/bin/env python3
import glob
import torch
import numpy as np
import json
from tqdm import tqdm
import sys
import argparse
from transformers import BertTokenizer, BertForMaskedLM, BertConfig,\
                         AlbertTokenizer, AlbertForMaskedLM, AlbertConfig
import os

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--type_of_model", default = 'albert', help = "pretrained LM type")
parser.add_argument("-p", "--path_to_pytorch_models", help = "path to pytorch_model")
parser.add_argument("--config_and_vocab", help = "path to config.json and vocab.model")
parser.add_argument("-s", "--step", type = str, help = "pretrained step")
parser.add_argument("-i", "--input", help = "path where you put your LAMA (駱馬) dataset")
parser.add_argument("-o", "--output", help = "output file")
args = parser.parse_args()

print("Knowledge prediction. step = ", args.step)

if args.type_of_model == 'albert':
  tokenizer = AlbertTokenizer(os.path.join(args.config_and_vocab, args.type_of_model, 'vocab.model'))
  config = AlbertConfig.from_json_file(os.path.join(args.config_and_vocab, args.type_of_model, 'config.json'))
  config.output_hidden_states = True
  model = AlbertForMaskedLM.from_pretrained(pretrained_model_name_or_path = None,
    config = config,
    state_dict = torch.load(os.path.join(
      args.path_to_pytorch_models, args.type_of_model, 'pytorch_model_' + args.step + '.bin')))
elif args.type_of_model == 'bert':
  tokenizer = BertTokenizer(os.path.join(args.config_and_vocab, args.type_of_model, 'vocab.model'))
  config = BertConfig.from_json_file(os.path.join(args.config_and_vocab, args.type_of_model, 'config.json'))
  config.output_hidden_states = True
  model = BertForMaskedLM.from_pretrained(pretrained_model_name_or_path = None,
    config = config,
    state_dict = torch.load(os.path.join(
      args.path_to_pytorch_models, args.type_of_model, 'pytorch_model_' + args.step + '.bin')))
else:
  raise NotImplementedError("The given model type %s is not supported" % args.type_of_model)

device = 'cuda' if torch.cuda.is_available else 'cpu'
model.eval().to(device)

rel_names = glob.glob(os.path.join(args.input, 'LAMA/TREx_processed/', '**/*.json'), recursive = True)

result = {}

for rel_name in rel_names:
  result[rel_name.split('/')[-1].split('.')[0]] = [0, 0]

for rel in tqdm(rel_names):
  f = open(rel, 'r')
  f_dict = json.load(f)
  f.close()

  for k, data in f_dict.items():
    tokens_tensor = torch.tensor([data['masked']])
    with torch.no_grad():
      _, prediction_scores = model(tokens_tensor.cuda(),masked_lm_labels = tokens_tensor.cuda())[:2]
    start = int(data['start'])
    end = int(data['end'])
    gt = data['unmasked'][start:end]
    prediction_scores = prediction_scores.squeeze().argmax(-1).cpu().numpy()[start:end]
    result[rel.split('/')[-1].split('.')[0]][1] += 1
    if np.all(np.equal(prediction_scores, gt)):
      result[rel.split('/')[-1].split('.')[0]][0] += 1

accuracy = []
x = ''
for rel in result:
  x += rel
  x += ','
  accuracy.append(str(result[rel][0] / (result[rel][1] + 1e-10)))
#print(x)  
accuracy = ','.join(accuracy)
accuracy = args.step + ',' + accuracy + '\n'
with open(os.path.join(args.output, args.type_of_model + '_knowledge_predict.txt'), 'a+') as f:
  f.write(accuracy)
