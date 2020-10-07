import torch
from transformers import BertTokenizer, BertModel, BertForPreTraining, WordpieceTokenizer, BertConfig,\
                         AlbertTokenizer, AlbertModel, AlbertForMaskedLM, AlbertConfig
import numpy as np
import sys
step = sys.argv[1]
tokenizer = AlbertTokenizer(vocab_file = '/work/dcml0714/albert/albert_base/30k-clean.model')
config = AlbertConfig.from_json_file('/work/dcml0714/albert/albert_base/albert_config.json')
config.output_hidden_states = True
config.output_attentions = True
model = AlbertForMaskedLM.from_pretrained(pretrained_model_name_or_path = None,
                                    config = config,
                                    state_dict = torch.load('/work/dcml0714/albert/pytorch_model/pytorch_model_' + step + '.bin'))
model.eval().cuda()

tag = {}
with open('/work/dcml0714/jiant_data/edges/ontonotes/const/pos/labels.txt') as f:
  while True:
    pos = f.readline().rstrip()
    if pos == "": break
    tag[pos] = np.asarray([0, 0])

import json
text_file = open('/work/dcml0714/jiant_data/edges/ontonotes/const/pos/conll-2012-test.json.retokenized.albert', 'r')

from tqdm import tqdm
for i, line in tqdm(enumerate(text_file.readlines())):
  data = json.loads(line)
  text = data['text'].split(' ')
  original_indexed_tokens = tokenizer.convert_tokens_to_ids(text)
  if data['targets'] == []:
    continue  
  #"""
  model_input = []
  for span in data['targets']:
    indexed_tokens = original_indexed_tokens.copy()
    for mask_pos in range(span['span1'][0], span['span1'][1]):
      indexed_tokens[mask_pos] = 4  

    indexed_tokens = [2] + indexed_tokens + [3]
    model_input.append(indexed_tokens)
  tokens_tensor = torch.tensor(model_input)

  with torch.no_grad():
    loss, prediction_scores = model(tokens_tensor.cuda(),masked_lm_labels = tokens_tensor.cuda())[:2]

  prediction_scores = prediction_scores.argmax(-1).cpu().numpy()[:, 1:-1]

  for i, span in enumerate(data['targets']):
    #print(text[span['span1'][0]: span['span1'][1]], span['label'])
    predict = prediction_scores[i][span['span1'][0]: span['span1'][1]]
    ground_truth = original_indexed_tokens[span['span1'][0]: span['span1'][1]]
    if np.all(np.equal(predict, ground_truth)):
      tag[span['label']][0] += 1
    tag[span['label']][1] += 1

  #"""
result = []
for key in tag:
  result.append(str(tag[key][0] / (tag[key][1] + 1e-10)))
result = ','.join(result)
result = step + ',' + result


with open('/home/dcml0714/albert_embryo/exp_result/pretrain/mask_predict.txt', 'a+') as f:
  f.write(result + '\n')  

