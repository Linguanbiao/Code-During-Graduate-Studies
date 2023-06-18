import numpy as np
import json
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

# sns.set(color_codes=True)
sns.set_theme(style='whitegrid')
# plt.figure(figsize=(12, 9), dpi=1080)
# sns.despine()
# sns.set_style({'font.family': ['Arial'],
#                'font.sans-serif': ['Arial']})
# sns.set_context({'axes.labelsize': 20,
#                  'axes.titlesize': 14,
#                  'xtick.labelsize': 14,
#                  'ytick.labelsize': 14,
#                  'legend.fontsize': 14,
#                  'lines.linewidth': 2})

with open('nodp_middle.json', 'r', encoding='utf8') as fp:
    json_data_15 = json.load(fp)
# #
with open('cifar10_middle_40.json', 'r', encoding='utf8') as fp:
    json_data_15n = json.load(fp)
# with open('2.0_middle.json', 'r', encoding='utf8') as fp:
#     json_data_110 = json.load(fp)
# with open('1_10_no.json', 'r', encoding='utf8') as fp:
#     json_data_110n = json.load(fp)
# with open('2_10.json', 'r', encoding='utf8') as fp:
#     json_data_210 = json.load(fp)
# with open('2_20.json', 'r', encoding='utf8') as fp:
#     json_data_220 = json.load(fp)
# with open('3_10.json', 'r', encoding='utf8') as fp:
#     json_data_310 = json.load(fp)
# with open('3_20.json', 'r', encoding='utf8') as fp:
#     json_data_320 = json.load(fp)
# #
# with open('log_cifar10_2.0_1.0.txt', 'r', encoding='utf8') as fp:
#     json_data_3 = json.load(fp)
# #
# with open('log_cifar10_2.0_1.0.txt', 'r', encoding='utf8') as fp:
#     json_data_4 = json.load(fp)
#
# with open('log_mnist2fashionmnist_3.0.txt', 'r', encoding='utf8') as fp:
#     json_data_5 = json.load(fp)
c15, c15n, c110, c110n, c210, c220, c310, c320 = [], [], [], [], [], [], [], []
epoch = [i for i in range(40)]

sum_ = 0
for idx, i in enumerate(json_data_15, start=1):
    sum_ += i
    if idx % 49 == 0:
        sum_ /= 49
        c15.append(sum_)
        if len(c15) == 40:
            break
        sum_ = 0

sum_ = 0
for idx, i in enumerate(json_data_15n, start=1):
    sum_ += i
    if idx % 49 == 0:
        sum_ /= 49
        c15n.append(sum_)
        if len(c15n) == 40:
            break
        sum_ = 0
# for i in json_data_110:
#     c110.append(i)
# for i in json_data_110n:
#     c110n.append(i)
# for i in json_data_210:
#     c210.append(i)
# for i in json_data_220:
#     c220.append(i)
# for i in json_data_310:
#     c310.append(i)
# for i in json_data_320:
#     c320.append(i)
# c1 += c2
# c1 += c3
# Dict['Epoch'] = epoch * 3
# Dict['Accuracy'] = c1
# threshold_ = [0.1] * 60
# threshold_ += [0.3] * 60
# threshold_ += [1.0] * 60
# Dict['Gradient clipping threshold'] = threshold_
# mnist
# parameter1 = [-3.75021130e-08, 6.93662977e-06,
#               -4.80450677e-04, 1.52979375e-02,
#               -2.15653935e-01, 1.34110972e+00]
# epoch_ = np.array(epoch)
# y1 = parameter1[0] * epoch_ ** 5 + \
#      parameter1[1] * epoch_ ** 4 + \
#      parameter1[2] * epoch_ ** 3 + \
#      parameter1[3] * epoch_ ** 2 + \
#      parameter1[4] * epoch_ ** 1 + \
#      parameter1[5]
# Dict['Gradient clipping threshold'] += y1.tolist()
# Dict['Type'] += ['B'] * 60
# fashionmnist
# parameter2 = [-2.00427215e-06, 4.77071240e-04,
#               -3.10446696e-02, 1.05305130e+00]
# epoch_ = np.array(epoch)
# y2 = parameter2[0] * epoch_ ** 3 + \
#      parameter2[1] * epoch_ ** 2 + \
#      parameter2[2] * epoch_ ** 1 + \
#      parameter2[3]
# Dict['Gradient clipping threshold'] += y2.tolist()
# Dict['Type'] += ['B'] * 60
#
# # cifar10
# parameter3 = [-2.23182262e-05, 2.46510724e-03,
#               -8.35273197e-02, 1.33081579e+00]
# epoch_ = np.array(epoch)
# y3 = parameter3[0] * epoch_ ** 3 + \
#      parameter3[1] * epoch_ ** 2 + \
#      parameter3[2] * epoch_ ** 1 + \
#      parameter3[3]
# Dict['Gradient clipping threshold'] += y3.tolist()
# Dict['Type'] += ['B'] * 60
# df = pd.DataFrame(Dict)
# print(df)
# #
# fig = sns.lineplot(data=df,
#                    x='Epoch',
#                    y='Accuracy',
#                    hue='Gradient clipping threshold',
#                    style='Gradient clipping threshold',
#                    palette=sns.color_palette("hls",3))
# # handles, labels = fig.get_legend_handles_labels()
# # print(handles, labels)
# fig.get_legend().set_title(None)
# # fig.legend(handles=handles[1:])
# pairplot_fig = fig.get_figure()
# pairplot_fig.savefig('acc_.pdf')
# for idx, i in enumerate(json_data_4.values()):
#     acc.append(i)
#
# for idx, i in enumerate(json_data_5.values()):
#     acc5.append(i)
# 'ro-', color='#4169E1'
# ax = plt.gca()
# x_major_locator = MultipleLocator(10)
# ax.xaxis.set_major_locator(x_major_locator)
plt.plot(epoch, c15, 'v:', color='red', alpha=0.8, linewidth=2, label='w\o dp')
plt.plot(epoch, c15n, 'o:', color='blue', alpha=0.8, linewidth=2, label='noise 0.91')
# plt.plot(epoch, c110, '-', color='orange', alpha=0.8, linewidth=2, label='noise 2.00')
# plt.plot(epoch, c110n, '-', color='blue', alpha=0.8, linewidth=1, label='noise 1.02')
# plt.plot(epoch, c210, '-', color='orange', alpha=0.8, linewidth=1, label='2 norm10')
# plt.plot(epoch, c220, '-', color='gold', alpha=0.8, linewidth=1, label='2 norm20')
# plt.plot(epoch, c310, '-', color='orange', alpha=0.8, linewidth=1, label='3 norm10')
# plt.plot(epoch, c320, '-', color='gold', alpha=0.8, linewidth=1, label='3 norm20')

# ax = plt.gca()
# x_major_locator = MultipleLocator(94)
# ax.xaxis.set_major_locator(x_major_locator)
# plt.xlim(0, 3000)
# my_label = [i for i in range(60)]
# plt.xticks(ticks=my_label, labels=my_label)
# plt.plot(epoch, c2,  color='blue', alpha=0.8, linewidth=1, label='old')
# plt.plot(epoch, acc3,  color='green', alpha=0.8, linewidth=1, label='0.5')
# plt.plot(epoch, acc4,  color='brown', alpha=0.8, linewidth=1, label='0.8')
# plt.plot(epoch, acc5,  color='black', alpha=0.8, linewidth=1, label='0.3')
plt.legend(loc="upper right")
plt.xlabel('Epoch')
plt.ylabel('L2 norm')
#
plt.savefig('all_middle_new.pdf')
# Dict = {}
# Dict['Epoch'] = []
# Dict['Accuracy'] = []
# Dict['Gradient clipping threshold'] = []
# EPOCHS = 60
# ACCS = 3
# threshold = [0.1, 0.3, 1.0]
# for epoch in range(EPOCHS):
#     for index in range(ACCS):
#         Dict['Epoch'].append(epoch)
#         Dict['Accuracy'].append(acc[index * 60 + epoch])
#         Dict['Gradient clipping threshold'].append(threshold[index])
# df = pd.DataFrame(Dict)
# print(df)
# fig = sns.lineplot(data=df,
#                    x='Epoch',
#                    y='Accuracy',
#                    hue='Gradient clipping threshold',
#                    style='Gradient clipping threshold',
#                    palette=sns.color_palette("Set2", 3))
# pairplot_fig = fig.get_figure()
# pairplot_fig.savefig('acc_.pdf', dpi=400)
