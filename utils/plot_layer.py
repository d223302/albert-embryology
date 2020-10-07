#!/usr/bin/env python3
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
#sns.palplot(sns.color_palette("Paired"))
sns.set(rc={'figure.figsize':(11.7,8.27)})
import pandas as pd
import glob
import sys
import os
data_path = sys.argv[1]
output_name = sys.argv[2]
data = [f for f in glob.glob(os.path.join(data_path + "**/*.txt"), recursive = True)]
print(data)
result = {}
for f in data: 
  layer = f.split('/')[-1].split('.')[0]
  f = pd.read_csv(f, header = None).to_numpy().squeeze()
  result[layer] = f

df = {}
#df['Pretrain step'] = np.arange(16) * 500
df['Pretrain step'] = np.arange(20) * 5000
for i in range(0, 12):
    df[str(i)] = result[str(i)][:20]
df = pd.DataFrame(df) 
df = df.melt('Pretrain step', var_name='layer',  value_name='pwcca coefficinet')
#df = df.melt('Finetune step', var_name='layer',  value_name='pwcca coefficinet')
ax = sns.lineplot(x="Pretrain step", y="pwcca coefficinet", hue='layer', data=df, legend = 'full' 
  )
"""
legend = []
#ax = plt.subplot(111)
for layer in range(1, 13):
  layer = str(layer)
  #if layer != '5': continue
  legend.append(layer)
  ax.plot(np.arange(len(result[layer])) * 10000, result[layer], alpha = 1)
  
ax.legend(legend, loc='lower right', ncol = 2, fontsize = 14)
ax.xaxis.set_major_formatter(matplotlib.ticker.EngFormatter())
ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(200000))
plt.xlabel('pretrain step', fontsize = 14)
plt.ylabel('mean cca', fontsize = 14)
"""
plt.savefig(output_name)  
plt.clf()
plt.close()
