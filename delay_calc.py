import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator

plt.rcParams['font.family'] = 'Times New Roman'

df = pd.read_csv('./datasets/wastewater.csv')
df = df.set_index('date')

v = 7130            # m3
sensor_loc = 3/4    # of tank
freq = 2            # min
delays = pd.DataFrame((sensor_loc * v) / ((df['IN_Q'].iloc[:]+0.1) / 60) / freq)
delays = delays.rename(columns={'IN_Q': 'Delay'})
delays['Delay'] = delays['Delay'][delays['Delay'] < 1000]

plt.figure(figsize=(6.5, 4))
ax = sns.histplot(x=delays['Delay'], bins=20)
    
counts, bins, _ = plt.hist(delays['Delay'], bins=20)
total = sum(counts)
for count, rect in zip(counts, ax.patches):
    height = rect.get_height()
    percentage = f'{100 * count / total:.1f}%'
    ax.annotate(percentage, xy=(1.05*rect.get_x() + rect.get_width() / 2, height), xytext=(0, 3),
                textcoords="offset points", ha='center', va='bottom', fontsize=5)
    
plt.xlabel(f'Delay with frequency of {freq} min', labelpad=5, fontsize=10)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.gca().xaxis.set_major_locator(MultipleLocator(60))
plt.show()
plt.savefig('delay_fig.pdf')