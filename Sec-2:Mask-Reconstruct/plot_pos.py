#!/usr/bin/env python
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import pandas as pd
import numpy as np
import matplotlib
import argparse
import seaborn as sns; sns.set()

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help = "Input data, should be csv formatted")
parser.add_argument("-o", "--output", help = "Output file name")
parser.add_argument("-v", '--verb', action = 'store_true', help = 'Plot verb only or not')
parser.add_argument("-r", "--rescale", action = 'store_true', help = "Whether to rescale the lines by the value of last pretraining step")
parser.add_argument("-m", "--max_data", type = int, default = -1, help = "Max data points to be plotted")
args = parser.parse_args()

colormap = plt.cm.gist_ncar
data = pd.read_csv(args.input)
data = data.sort_values(by=['step'])
legend = data.columns
if not args.verb:
  target = ['CC', 'DT', 'IN', 'JJ', 'NN', 'NNP', 'PRP', 'RB', 'VB']
  legend_full = ['conj.', 'det.', 'prep.', 'adj.', 'noun', 'proper noun', 'pron.', 'adv.', 'verb']
else:
  target = ['VB', 'VBD', 'VBG', 'VBN', 'VBZ']
  legend_full = ['V', 'V-ed', 'V-ing', 'V-en', 'V-es']

data = data.values.T
ax = plt.subplot(111)
step = data.shape[1]
for i in range(1, len(legend)):
  if legend[i] in target:
    if args.rescale: 
      ax.plot(data[0][:args.max_data], data[i][:args.max_data] / data[i][-1], alpha = 0.7)
    else:
      ax.plot(data[0][:args.max_data], data[i][:args.max_data], alpha = 0.7)  
box = ax.get_position()
# Put a legend to the right of the current axis
ax.legend(legend_full, loc='lower right', ncol = 2, fontsize = 12)
ax.xaxis.set_major_formatter(matplotlib.ticker.EngFormatter())
ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(10000))
plt.xlabel('pretrain step', fontsize = 14)
if args.rescale: 
  plt.ylabel('accuracy (rescaled)', fontsize = 14)
else:
  plt.ylabel('accuracy', fontsize = 14)
plt.savefig(args.output)  
plt.clf()
plt.close()

