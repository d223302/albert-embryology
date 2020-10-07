#!/usr/bin/env python
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns; sns.set()
import pandas as pd
import numpy as np
colormap = plt.cm.gist_ncar
data = pd.read_csv('bert_knowledge_predict.txt')
data = data.sort_values(by=['step'])
legend = data.columns
#target = ['P103', 'P276', 'P131', 'P36', 'P20', 'P176', 'P127', 'P364', 'P19']
#target = ['P103', 'P407', 'P364', 'P176', 'P449', 'P30', 'P138'] 
#target = ['P449', 'P159', 'P407', 'P176', 'P364', 'P103', 'P1376']
#target = ['P140', 'P103', 'P176', 'P138', 'P407', 'P1376', 'P159']
target = legend
line_style = ['-', '--', ':']
temp = []
plt.figure(figsize = (8, 6))
data = data.values.T
print(data.shape)
ax = plt.subplot(111)
step = data.shape[1]
for i in range(1, len(legend)):
  if legend[i] in target:  
    ax.plot(data[0], data[i], linestyle = line_style[i // 10])
    #ax.plot(data[0], data[i])
    temp.append(legend[i])
ax.xaxis.set_major_formatter(matplotlib.ticker.EngFormatter())
ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(200000))
ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.1))
# Put a legend to the right of the current axis
ax.legend(temp, loc='upper left', ncol = 4)
#ax.legend(temp, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.ylim(0, 1)
plt.xlabel('Pretrain steps', fontsize = 16)
plt.ylabel('Accuracy', fontsize = 16)
plt.savefig('knowledge_last.pdf')  
plt.clf()
plt.close()

