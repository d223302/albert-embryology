#!/usr/bin/env python3
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, trange
import numpy as np
import pickle
import argparse
import os
import copy
from sklearn.metrics import f1_score

class ProbeData(Dataset):
  def __init__(self, data_dir):
    super(ProbeData, self).__init__()
    print("Loading ", data_dir)
    with open(data_dir, 'rb') as f:
      self.data = pickle.load(f)

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    label = self.data[idx][0]
    span1 = torch.from_numpy(self.data[idx][1])
    try:
      span2 = torch.from_numpy(self.data[idx][2])
    except IndexError:
      span2 = None      
    if span2 is not None:
      return label, span1, span2
    else:
      return label, span1

class LinearProbe(nn.Module):
  def __init__(self, output_classes, span2 = False, input_dim = 768):
    super(LinearProbe, self).__init__()
    if span2:
      input_dim *= 2
    self.model = nn.Linear(input_dim, output_classes)

  def forward(self, span1, span2 = None):
    if span2 is not None:
      feat = torch.cat((span1, span2), -1)
    else:
      feat = span1
    return self.model(feat)

parser = argparse.ArgumentParser()
parser.add_argument('-t', "--train_dir", help = "training_feature file")
parser.add_argument('-d', "--dev_dir", help = "testing data dir")
parser.add_argument('-g', "--test_dir", help = "test data dir")
parser.add_argument('-o', "--output", default = None, help = "file in which result will be printed to, defalut is std out")
parser.add_argument('-b', "--batch_size", type = int, default = 1024)
parser.add_argument('-l', "--lr", type = float, default = 1e-4)
parser.add_argument('-s', "--max_step", type = int, default = 50000)
parser.add_argument('-w', "--num_of_workers", type = int, default = 32)
parser.add_argument("--span2", action = "store_true")
parser.add_argument('-c', "--classes", type = int, help = "total number of labels")
parser.add_argument('-e', "--eval_step", type = int, default = 1000, help = "evaluate every e step")
parser.add_argument('-p', '--pretrain_step', type = int, help = "Pretrain step")
#parser.add_argument('--layer', type = int, help = 'layer from which feature is drawn')
args = parser.parse_args()



train_loader = DataLoader(ProbeData(args.train_dir),
    batch_size = args.batch_size,
    num_workers = args.num_of_workers,
    shuffle = True)
dev_loader = DataLoader(ProbeData(args.dev_dir),
    batch_size = args.batch_size,
    num_workers = args.num_of_workers,
    shuffle = False)
test_loader = DataLoader(ProbeData(args.test_dir),
    batch_size = args.batch_size,
    num_workers = args.num_of_workers,
    shuffle = False)

print("Training examples: ", len(train_loader))
print("Validation examples: ", len(dev_loader))

train_iter = iter(train_loader)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = LinearProbe(args.classes, args.span2)
model.to(device)
lr = args.lr
optimizer = torch.optim.Adam(model.parameters(), lr)
CE_loss = nn.CrossEntropyLoss().to(device)

t = tqdm(range(args.max_step), desc =  'Training loss', leave = True)
f1_result = []
for step in t:
  model.train()  
  try:
    feat = train_iter.next()
  except StopIteration:
    train_iter = iter(train_loader)
    feat = train_iter.next()
  label = feat[0].to(device)
  s1 = feat[1].to(device)
  if args.span2:
    s2 = feat[2].to(device)
  else:
    s2 = None
  
  logits = model(s1, s2)
  optimizer.zero_grad()
  loss = CE_loss(logits, label)

  loss.backward()
  torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
  optimizer.step()
  t.set_description("Loss: %f" % loss.item())
  t.refresh()

  
  if step % args.eval_step == 0:
    model.eval()  
    gt = []
    pred = []
    for feat in dev_loader:
      label = feat[0].numpy()
      s1 = feat[1].to(device)
      if args.span2:
        s2 = feat[2].to(device)
      else:
        s2 = None
      batch_pred = model(s1, s2).argmax(-1).detach().cpu().numpy()

      gt.extend(label)
      pred.extend(batch_pred)
    f1 = f1_score(gt, pred, average = 'micro')
    if step == 0:
      f1_result = [f1]  
    if f1 > f1_result[-1]:
      f1_result = [f1]
      best_model = copy.deepcopy(model)
    else:  
      f1_result.append(f1)
    print("\nStep ", step, ", development set micro f1 score: ", f1)
  if len(f1_result) > 5:
    lr /= 2  
    optimizer = torch.optim.Adam(model.parameters(), lr)  
  if len(f1_result) > 20:  
    model = best_model  
    break


print("Evaluating trained model on test set (", len(test_loader), ") examples")
model.eval()  
gt = []
pred = []
for feat in test_loader:
  label = feat[0].numpy()
  s1 = feat[1].to(device)
  if args.span2:
    s2 = feat[2].to(device)
  else:
    s2 = None
  batch_pred = model(s1, s2).argmax(-1).detach().cpu().numpy()

gt.extend(label)
pred.extend(batch_pred)
f1 = f1_score(gt, pred, average = 'micro')
print("Test set micro f1 score: ", f1)


with open(args.output, 'a+') as f:
  result = str(args.pretrain_step) + ',' + str(f1) + '\n'  
  f.write(result)
