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
parser.add_argument("-p", "--path_to_pytorch_models", help = "path to pytorch_model")
parser.add_argument("--config_and_vocab", help = "path to config.json and vocab.model")
parser.add_argument("-s", "--step", type = str, help = "pretrained step")
parser.add_argument("-d", "--data", help = "path where you put your processed ontonotes data")
parser.add_argument("-o", "--output", help = "output file")
args = parser.parse_args()
print("Reconstruction. step = ", args.step)
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

tag = {}
with open(os.path.join(args.data, 'ontonotes/const/pos/labels.txt')) as f:
  while True:
    pos = f.readline().rstrip()
    if pos == "": break
    tag[pos] = np.asarray([0, 0])

    
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
  model_inputs = []
  for span in labels:
    span1 = [] 
    indexed_tokens = retokenized.copy()
    for pos in range(span['span1'][0], span['span1'][1]):
      select = np.where(np.asarray(re_2_o) == pos)[0]
      span1.extend(select)
    for i in span1:
      indexed_tokens[i] = tokenizer.mask_token_id 
    indexed_tokens = [tokenizer.cls_token_id] + indexed_tokens + [tokenizer.sep_token_id]
    model_inputs.append(indexed_tokens)

  retokenized.insert(0, tokenizer.cls_token_id)
  retokenized.append(tokenizer.sep_token_id)
  mlm_labels = torch.tensor([retokenized]).repeat(len(labels), 1).to(device)
  token_tensors = torch.tensor(model_inputs).to(device)
  prediction_scores = ()
  if len(labels) == 0: 
    continue
  for i in range((len(labels) - 1) // 16 + 1):
    with torch.no_grad():
      loss, batch_prediction_scores = model(token_tensors[i * 16: (i + 1) * 16], masked_lm_labels = mlm_labels[i * 16: (i + 1) * 16])[:2]
    prediction_scores += (batch_prediction_scores.argmax(-1).cpu().numpy()[:, 1:-1],)
  indexed_tokens = mlm_labels[0, 1:-1].cpu().numpy()
  prediction_scores = np.concatenate(prediction_scores, 0)

  for i, span in enumerate(labels):
    span1 = []
    for pos in range(span['span1'][0], span['span1'][1]):
      select = np.where(np.asarray(re_2_o) == pos)[0]
      span1.extend(select)
    predict = prediction_scores[i][span1]
    ground_truth = indexed_tokens[span1]
    tokenized_len = str(len(predict))
    if np.all(np.equal(predict, ground_truth)):
      tag[span['label']][0] += 1
    tag[span['label']][1] += 1

result = []
for key in tag:
  result.append(str(tag[key][0] / (tag[key][1] + 1e-10)))
result = ','.join(result)
result = args.step + ',' + result


with open(os.path.join(args.output, args.type_of_model + "_mask_predict.txt"), 'a+') as f:
  f.write(result + '\n')  
