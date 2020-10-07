#!/usr/bin/env python3
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib

current_palette = sns.color_palette()

srl = pd.read_csv('srl.txt', sep = ',', header = None).values.T
pos = pd.read_csv('pos.txt', sep = ',', header = None).values.T
const = pd.read_csv('const.txt', sep = ',', header = None).values.T
coref = pd.read_csv('coref.txt', sep = ',', header = None).values.T
m = 40
fig, ax2 = plt.subplots()
ax2.set_xlabel('Pretrain steps')
ax2.set_ylabel('Evaluation Result', color = 'black')
ax2.plot(srl[0][:m], srl[1][:m], color=current_palette[2], label = 'SRL')
ax2.plot(pos[0][:m], pos[1][:m], color=current_palette[3], label = 'POS')
ax2.plot(const[0][:m], const[1][:m], color=current_palette[4], label = 'Const')
ax2.plot(coref[0][:m], coref[1][:m], color=current_palette[5], label = 'Coref')
ax2.set_ylim(0.4, 1)
ax2.xaxis.set_major_formatter(matplotlib.ticker.EngFormatter())
ax2.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(30000))
ax2.legend(title = 'Evaluation result', ncol = 2, loc = 'lower right')
ax2.tick_params(axis='y', labelcolor='black')


#fig.tight_layout()
#"""
plt.savefig('electra_pretrain_150k.pdf')
