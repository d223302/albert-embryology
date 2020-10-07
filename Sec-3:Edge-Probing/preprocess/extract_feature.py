#!/usr/bin/env python3
import json
import transformers
from transformers import (BertTokenizer, BertModel, BertConfig,
                          AlbertModel, AlbertConfig, AlbertTokenizer,
                          ElectraConfig, ElectraTokenizer, ElectraModel,
                          )
import numpy as np
import torch
import argparse
import os
from multiprocessing import Pool
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle

def collate_fn(batch):
  batch.sort(key=lambda x: x[0].shape[0] , reverse=True)
  input_ids, re_2_o, attn_mask, labels = zip(*batch)
  input_ids = pad_sequence(input_ids, True)
  attn_mask = pad_sequence(attn_mask, True)
  return input_ids, attn_mask, re_2_o, labels

class Sentence(Dataset):
  def __init__(self, data):
    self.data = data
  def __len__(self):
    return len(self.data)
  def __getitem__(self, idx):
    input_ids, re_2_o, labels = self.data[idx]
    return input_ids, np.asarray(re_2_o), torch.ones_like(input_ids).float(), labels


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("-m", "--model", type = str, help = "LM used to extract features")
  parser.add_argument("-i", "--input", type = str, help = ".json file to be retokenized")
  parser.add_argument("-o", "--output", type = str, help = "output feature directory")
  parser.add_argument("-b", "--batchsize", type = int, default = 32, 
    help = "batch size whene forwarding through language model")
  parser.add_argument('--layer', type = int, help = "layer from which feature should be extracted")
  parser.add_argument("-c", "--config", type = str, help = "config for model")
  parser.add_argument("-t", "--tokenizer", type = str, help = "path to vocab file for tokenizer")
  parser.add_argument("-l", "--label", type = str, help = "Path to label.txt")
  parser.add_argument("-p", "--probe", type = str, help = "Probed task name")
  parser.add_argument("--data_type", type = str, help = "train, dev, or test set")
  args = parser.parse_args()

  label_dict = {}
  with open(args.label, 'r') as f:
    line_id = 0  
    while True:
      line = f.readline()
      if line == '': break
      line = line.rstrip()
      label_dict[line] = line_id
      line_id += 1
      
  if 'albert' in args.model:
    model_type = 'albert'
    tokenizer = AlbertTokenizer(vocab_file = args.tokenizer)
    config = AlbertConfig.from_json_file(args.config)
    model = AlbertModel.from_pretrained(pretrained_model_name_or_path = None,
      config = config,
      state_dict = torch.load(args.model))
  elif 'bert' in args.model:
    model_type = 'bert'
    tokenizer = BertTokenizer(vocab_file = args.tokenizer)
    config = BertConfig.from_json_file(args.config)
    model = BertModel.from_pretrained(pretrained_model_name_or_path = None,
      config = config,
      state_dict = torch.load(args.model))
  elif 'electra' in args.model:
    model_type = 'electra'
    tokenizer = ElectraTokenizer(vocab_file = args.tokenizer)
    config = ElectraConfig.from_json_file(args.config)
    model = ElectraModel.from_pretrained(pretrained_model_name_or_path = None,
      config = config,
      state_dict = torch.load(args.model))
  else:
    raise NotImplementedError("The model is currently not supported")
  
  def process_line(line):
    data = json.loads(line)
    tokens = data['text'].split(' ')
    labels = data['targets']
    return tokens, labels

  def retokenize(tokens_labels):
    tokens, labels = tokens_labels  
    retokenized = []
    re_2_o = [] # same length as retokenized sequence, store the mapping of index from retokenized to original seq
    for word_id, token in enumerate(tokens):
      token = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token))
      retokenized.extend(token)
      re_2_o.extend([word_id for _ in range(len(token))])
    retokenized.insert(0, tokenizer.cls_token_id)
    retokenized.append(tokenizer.sep_token_id)
    input_ids = torch.tensor(retokenized)
    return input_ids, re_2_o, labels

  pool = Pool(4)
  with open(args.input, 'r') as f:
    processed_data = pool.map(process_line, f)
  pool.close()
  pool.join()
  #print(len(processed_data))

  processed_data = list(map(retokenize, processed_data))

  print("Total number of sentences: ", len(processed_data))
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  model.eval().to(device)

  result = []

  dataloader = torch.utils.data.DataLoader(Sentence(processed_data),
    batch_size = args.batchsize, 
    num_workers = 32,
    collate_fn = collate_fn)

  def post_process(fn_input):
    feature = []  
    model_output, re_2_o, labels, seq_len, input_ids = fn_input
    model_output = model_output[:seq_len][1:-1]
    for target in labels:
      span1 = []
      for pos in range(target['span1'][0], target['span1'][1]):
        select = np.where(np.asarray(re_2_o) == pos)[0]
        span1.extend(model_output[select])
      span1 = np.stack(span1).mean(0)
#     print(span1.shape)
      if 'span2' in target.keys():
        span2 = []
        for pos in range(target['span2'][0], target['span2'][1]):
          select = np.where(np.asarray(re_2_o) == pos)[0]
          span2.extend(model_output[select])
        span2 = np.stack(span2).mean(0)
      else:
        span2 = None
#      print(target['label'])  
      label = label_dict[target['label']]
      if span2 is not None:
        feature.append([label, span1, span2])
      else:
        feature.append([label, span1])
    return feature


  for input_ids, attn_mask, re_2_o, labels in tqdm(dataloader):
    with torch.no_grad():
      model_output = model(input_ids.to(device), 
        attention_mask = attn_mask.to(device),
        output_attentions = True, 
        output_hidden_states = True)
      if model_type == 'electra':
        model_output = model_output[1][args.layer].detach().cpu().numpy() 
      else:
        model_output = model_output[2][args.layer].detach().cpu().numpy()
    seq_len = attn_mask.sum(-1).cpu().long().numpy()
    map_input = [*zip(model_output, re_2_o, labels, seq_len, input_ids)]
    for x in map(post_process, map_input):
      result.extend(x)

  #processed_data = list(map(post_process, result))
  processed_data = result
  step = args.model.split('_')[-1].split('.')[0]
  output_dir = os.path.join(args.output, args.probe + '-' + args.data_type + '-' + model_type + '-' + step + '.pkl')
  with open(output_dir, 'wb') as f:
    pickle.dump(processed_data, f)
  
