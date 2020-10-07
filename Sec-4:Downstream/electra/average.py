#!/usr/bin/env python3
import numpy as np
import pandas as pd
sst = (pd.read_csv('SST-2.txt', header = None).values).astype(float)
sts = (pd.read_csv('STS-B.txt', header = None).values).astype(float)
mnli = (pd.read_csv('MNLI.txt', header = None).values).astype(float)
rte = (pd.read_csv('RTE.txt', header = None).values).astype(float)
mrpc = (pd.read_csv('MRPC.txt', header = None).values).astype(float)
qnli = (pd.read_csv('QNLI.txt', header = None).values).astype(float)
cola = (pd.read_csv('CoLA.txt', header = None).values).astype(float)
qqp = (pd.read_csv('QQP.txt', header = None).values).astype(float)
data = {}
for x in sst, sts, mnli, rte, mrpc, qnli, cola, qqp:
  for i in range(x.shape[0]):
    if str(x[i][0]) not in data:
      data[str(x[i][0])] = []
    data[str(x[i][0])].append(np.mean(x[i][1:]))

with open('GLUE.txt', 'w') as gl:
    for k, v in data.items():
      y = k + ',' + str(np.mean(v)) + '\n'
      gl.write(y)
