import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = {}
data['epoch'] = []
data['flat'] = []
data['acc'] = []

sigma = '1.56'
flat = ['0.5', '1.2']

for i in range(1, 6):
    for flat_ in flat:
        filename = 'sigma_{}_epoch_40_flat_{}_{}.npy'.format(sigma, flat_, i)
        file = np.load(filename, allow_pickle=True)
        file_ = file.item()
        for k in file_:
            data['epoch'].append(k)
            data['flat'].append(flat_)
            data['acc'].append(file_[k])

for i in range(1, 5):
    filename = 'sigma_{}_epoch_40_dec_{}_{}.npy'.format('1.56', 2.5, i)
    file = np.load(filename, allow_pickle=True)
    file_ = file.item()
    for k in file_:
        data['epoch'].append(k)
        data['flat'].append('dec')
        data['acc'].append(file_[k])

df = pd.DataFrame(data)

sns.set_theme(style='whitegrid')
sns.lineplot(data=df, x='epoch', y='acc', hue='flat')
plt.savefig('1.pdf')