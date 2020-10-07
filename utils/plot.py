#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help = "Input path of .txt files")
parser.add_argument('-o', '--output', help = "Output file name")
args = parser.parse_args()

files = [f for f in glob.glob(os.path.join(args.input, "**/*.txt"), recursive = True)]
legend = []
for f in files:
  x = pd.read_csv(f, header = None).to_numpy()
  x = x[x[:, 0].argsort()]#[:16]
  layer = f.split('/')[-1].split('.')[0]
  plt.plot(x[:, 0], x[:, 1])
  legend.append(layer)

plt.legend(legend)
plt.savefig(args.output)
