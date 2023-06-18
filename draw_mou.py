from matplotlib import pyplot as plt
import numpy as np
import json
import seaborn as sns

sns.set_theme(style='whitegrid')

mou = [0.0, 0.1, 0.3 , 0.5 ,0.7, 0.9 ]
# color = ['#DA70D6', '#FFA500','#4B0082' , '#6495ED' ,'#7B68EE' ,'#CD5C5C','#CCCC33' ,'#990000','#666600','#990099']
color = ['#DA70D6', '#FFA500','#4B0082' , '#6495ED' ,'#7B68EE' ,'#CD5C5C']
for idx, m in enumerate(mou):
    # if(idx != 5) : 
    #     filename = 'imagenette_mou/tanh/eps_11.0/alexnet_imagenette_lr0.001_momentum{}_batchsize64_eps_11_c6_tanh.json'.format(str(m))
    # else :
    filename = '/kolla/lgb/dynamic_momentum/fashionmnist/no_private/m_{}.json'.format(str(m))
    acc = []
    epoch = [i for i in range(40)]
    with open(filename, 'r', encoding='utf8') as fp:
        json_data = json.load(fp)
    for i in epoch:
        acc.append(json_data[str(i)])
    plt.plot(epoch, acc, color=color[idx], alpha=0.8, linewidth=2, label='m = {}'.format(str(m)))

plt.legend(loc="upper left")
plt.xlabel('Epoch')
plt.ylabel('Acc')
plt.savefig('Z_DP_momentum_fingures/fashionmnist/no_private_2.png'.format(str(m)))
    