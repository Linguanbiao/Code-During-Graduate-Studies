import seaborn as sns
import matplotlib.pyplot as plt
import json
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

sns.set_style('whitegrid')
fig, ax = plt.subplots(2, 2, figsize=(30, 30))
fig.subplots_adjust(left=0, right=1, wspace=0.1)
__method__ = ['w/o', '0.1', '0.3', '0.5', '0.7', '0.9', '0.93', '0.95', '0.97', '0.99']
__dataset__ = ['mnist', 'fashionmnist', 'cifar10', 'imagenette']
__title__ = {'mnist': '(a) MNIST', 'fashionmnist': '(b) Fashion-MNIST',
             'cifar10': '(c) CIFAR-10', 'imagenette': '(d) Imagenette'}
# __color__ = {'nodp':'#023EFF', 'dpsgd_relu':'#FF7C00', 'yuda':'#1AC938', 'adaclip':'#E8000B', 'ada':'#8B2BE2', 'dynamic':'#9F4800', 'papernot':'#F14CC1', 'transfer':'#A3A3A3', 'dec':'#FFC400'}
__color__ = {'w/o': '#023EFF', '0.1': '#FF7C00', '0.3': '#1AC938', '0.5': '#E8000B', '0.7': '#8B2BE2',
             '0.9': '#FFCC33', '0.93': '#F14CC1', '0.95': '#A3A3A3', '0.97': '#008c8c', '0.99': '#FFCCCC'}
for i in range(4):
    data = {}
    data['method'] = []
    data['acc'] = []
    data['epoch'] = []

    filelist = json.load(open('file2.json'))
    dataset = __dataset__[i]
    methods = filelist[dataset]

    for name in __method__:
        result_list = methods[name]
        for name_ in result_list:
            with open(name_) as f:
                json_data = json.load(f)
            for epoch in json_data:
                data['method'].append(str(name))
                data['epoch'].append(int(epoch))
                data['acc'].append(float(json_data[epoch]))

    df = pd.DataFrame(data)

    if i == 2:
        sns.lineplot(data=df, ax=ax[1][i % 2], x='epoch', y='acc', hue='method', linewidth=2.7, palette=__color__)
    else:
        if i > 1:
            sns.lineplot(data=df, ax=ax[1][i % 2], x='epoch', y='acc', hue='method',
                         linewidth=2.7, palette=__color__, legend=False)
        else:
            sns.lineplot(data=df, ax=ax[0][i], x='epoch', y='acc', hue='method',
                         linewidth=2.7, palette=__color__, legend=False)

    if i == 3:
        xlabels = [i for i in range(0, 31, 10)]
    else:
        xlabels = [i for i in range(0, 41, 10)]
    if i == 0 or i == 1:
        ax[0][i].set_xticks(xlabels)
        ax[0][i].tick_params(axis='both', labelsize=25)
    else:
        ax[1][i % 2].set_xticks(xlabels)
        ax[1][i % 2].tick_params(axis='both', labelsize=25)

    if i == 0:
        ax[0][i].set_ylabel('Accuracy', fontsize=30)
        ax[0][i].set_xlabel('Epoch', fontsize=30)
    elif i == 2:
        ax[1][i % 2].set_ylabel('Accuracy', fontsize=30)
        ax[1][i % 2].set_xlabel('Epoch', fontsize=30)
    else:
        if i < 2:
            ax[0][i].set_ylabel('')
            ax[0][i].set_xlabel('Epoch', fontsize=30)
        else:
            ax[1][i % 2].set_ylabel('')
            ax[1][i % 2].set_xlabel('Epoch', fontsize=30)

    if i == 0:
        ax[0][i].set_ylim(0.05, 1)
    elif i == 1:
        ax[0][i].set_ylim(0.05, 0.95)
    elif i == 2:
        ax[1][i % 2].set_ylim(0.05, 0.7)
    elif i == 3:
        ax[1][i % 2].set_ylim(0.05, 0.6)

    if i == 0 or i == 1:
        ax[0][i].set_title(__title__[dataset], fontsize=30)
    else:
        ax[1][i % 2].set_title(__title__[dataset], fontsize=30)

    if i == 2:
        handles, labels = ax[1][i % 2].get_legend_handles_labels()
        orders = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        handles = [handles[i] for i in orders]
        labels_ = [labels[i] for i in orders]
        ax[1][i % 2].legend(loc=2, bbox_to_anchor=(0.05, -0.1), borderaxespad=1, ncol=10, handlelength=5, columnspacing=10, frameon=False, markerscale=20,
                            handles=handles, labels=['w/o', '0.1', '0.3', '0.5', '0.7', '0.9', '0.93', '0.95', '0.97', '0.99'])
        plt.setp(ax[1][i % 2].get_legend().get_texts(), fontsize=30)

    # if i == 0:
    #     axins = ax[0][i].inset_axes((0.3, 0.1, 0.4, 0.4))
    #     axins.set_ylim(0.97, 1.0)
    #     axins.set_xlim(35, 39)
    #     axins.tick_params(axis='both',labelsize=20)
    #     sns.lineplot(data=df, ax=axins, x='epoch', y='acc', hue='method', linewidth=2.7, palette=__color__, legend=False)
    #     axins.set_ylabel('')
    #     axins.set_xlabel('')
    #     mark_inset(ax[0][i], axins, loc1=2, loc2=4, fc='none', ec='#696969')
    # elif i == 1:
    #     axins = ax[0][i].inset_axes((0.4, 0.07, 0.4, 0.4))
    #     axins.set_ylim(0.87, 0.92)
    #     axins.set_xlim(35, 39)
    #     axins.tick_params(axis='both',labelsize=20)
    #     sns.lineplot(data=df, ax=axins, x='epoch', y='acc', hue='method', linewidth=2.7, palette=__color__, legend=False)
    #     axins.set_ylabel('')
    #     axins.set_xlabel('')
    #     mark_inset(ax[0][i], axins, loc1=2, loc2=4, fc='none', ec='#696969')
    # elif i == 2:
    #     axins = ax[1][i % 2].inset_axes((0.4, 0.07, 0.4, 0.4))
    #     axins.set_ylim(0.72, 0.79)
    #     axins.set_xlim(35, 39)
    #     axins.tick_params(axis='both',labelsize=20)
    #     sns.lineplot(data=df, ax=axins, x='epoch', y='acc', hue='method', linewidth=2.7, palette=__color__, legend=False)
    #     axins.set_ylabel('')
    #     axins.set_xlabel('')
    #     mark_inset(ax[1][i % 2], axins, loc1=2, loc2=4, fc='none', ec='#696969')


plt.savefig('dpsgd_all_acc.pdf', bbox_inches='tight', pad_inches=0.2)
