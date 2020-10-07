#!/usr/bin/env python3
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
import sys
import glob 
import pandas as pd


if __name__ == '__main__':
  ###
  ###  Usage: python3 plot.py  PATH_TO_DATA_DIR OUTPUT_FILE_NAME TITLE_NAME 
  ###  Data dir contains txt file with file name as legend and.
  ###  first col should be x axis, second col should be y axis, seperated by ','
  ### TODO: add max length
  data = {}
  #albert = {}
  #bert = {}
  current_palette = sns.color_palette()
  txt = [f for f in glob.glob(sys.argv[1] + "**/*.txt", recursive = True)]
  legend = [] 
  for i, f in enumerate(txt): 
    if 'SQ' in f:
      txt.remove(f)
      txt.append(f)
      break
  for f in txt:
    field_name = f.rstrip().split('/')[-1].split('.')[0]
    if 'SQ' in field_name:
      field_name = 'SQuAD2.0'  
    temp = pd.read_csv(f, sep = ',', header = None).values
    #albert[field_name] = temp[-2, -1]
    #bert[field_name] = temp[-1, -1]
    temp = (temp).astype(float)
    temp = temp[temp[:, 0].argsort()]
    legend.append(field_name)
    #print(temp.shape)
    data[field_name] = temp

  fig = plt.figure(figsize=(8, 4.5))
  ax = fig.add_subplot(1, 1, 1)
  for i, key in enumerate(data): 
      best_step = data[key].T[0][np.argmax(data[key].T[1])]
      if 'SQ' in legend[i]:
        max_f1 = np.max(data[key].T[1] + data[key].T[2])
        ax.plot(data[key].T[0], (data[key].T[1] + data[key].T[2]) / 2, alpha = 0.8,
              markevery = list(np.equal((data[key].T[1] + data[key].T[2]), max_f1)),
              marker = '.',
              markersize = 3.5,
              label = legend[i])
      elif 'MNLI' in legend[i]: 
        max_f1 = np.max(data[key].T[1] + data[key].T[2])
        ax.plot(data[key].T[0], (data[key].T[1] + data[key].T[2]) * 50, alpha = 0.8,
              markevery = list(np.equal((data[key].T[1] + data[key].T[2]), max_f1)),
              marker = '.',
              markersize = 3.5,
              label = legend[i])
      else: 
        max_f1 = np.max(data[key].T[1])
        ax.plot(data[key].T[0], data[key].T[1] * 100, alpha = 0.8,
              markevery = list(np.equal(data[key].T[1], max_f1)),
              marker = '.',
              markersize = 3.5,
              label = legend[i])
      #ax.plot(best_step, albert[key] * 100, c = current_palette[i], marker = '+', ms = 3.5, alpha = 1.0)
      #ax.plot(best_step, bert[key] * 100, c = current_palette[i], marker = 'x', ms = 3.5, alpha = 1.0)
  ax.legend(ncol = 3, loc = 'lower right')
  ax.xaxis.set_major_formatter(matplotlib.ticker.EngFormatter())
  ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(200000))
  plt.xlabel('Pretrain steps')
  plt.ylabel('Evaluation result')
  plt.ylim(50, 90)
  ax.grid(b=True, which='major', color='w', linestyle='-') 
  plt.savefig('glue.pdf')
  plt.clf()
  plt.close()



