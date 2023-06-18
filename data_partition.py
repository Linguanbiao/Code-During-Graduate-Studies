# ！-*- coding:utf-8 -*-
import random

import numpy as np


def clipDataTopX(dataToClip, top=3):
    res = [sorted(s, reverse=True)[0:top] for s in dataToClip]
    return np.array(res)


def shuffleAndSplitData(dataX, dataY, cluster):
    # dataX = np.concatenate((dataX, testX), axis = 0)
    # dataY = np.concatenate((dataY, testY), axis = 0)

   # shuffle data
    c = list(zip(dataX, dataY))
    random.shuffle(c)
    dataX, dataY = zip(*c)

    # cluster_10000 = int(cluster / 2 + cluster)

    toTrainData = np.array(dataX[:cluster])
    toTrainLabel = np.array(dataY[:cluster])

    toTestData = np.array(dataX[cluster:cluster * 2])
    toTestLabel = np.array(dataY[cluster:cluster * 2])
    # toTestData = np.array(dataX[cluster:cluster_10000])
    # toTestLabel = np.array(dataY[cluster:cluster_10000])

    # cluster_10000_ = cluster_10000 + cluster
    # shadowData = np.array(dataX[cluster_10000:cluster_10000_])
    # shadowLabel = np.array(dataY[cluster_10000:cluster_10000_])

    shadowData = np.array(dataX[cluster * 2:cluster * 3])
    shadowLabel = np.array(dataY[cluster * 2:cluster * 3])

    shadowTestData = np.array(dataX[cluster * 3:cluster * 4])
    shadowTestLabel = np.array(dataY[cluster * 3:cluster * 4])
    # shadowTestData = np.array(dataX[cluster_10000_:cluster*3])
    # shadowTestLabel = np.array(dataY[cluster_10000_:cluster*3])

    return toTrainData, toTrainLabel, shadowData, shadowLabel, toTestData, toTestLabel, shadowTestData, shadowTestLabel
# import random
# boys_name_list = ['知与谁','黎志浩','梁家铭','陈畅','李聪','李颜','蔡贺秋',
#                  '张杰华','莫康华','刘贻钧','Liam','RillaLeung','姚家欣','李兴宇',
#                  '0与1的邂逅','邓伟初','末疚','李键鸿','周怀成','黄鹏飞','彭嘉豪','huangteng',
#                  '林观彪','张镇鑫','朱陶宇','王宇峰','彭诗煜','李浪','侯瑞涛']
# girls_name_list = ['胡丽','杨慧','任华丽','张盈盈','罗紫丹','李丹','揭晚晴','李文君','闫红洋']
# need_dic = {'男女混唱1':[],'男女混唱2':[],'个人演出1':[],'个人演出2':[]}
# keep_choose = []  # 出现过一次的选择不再选择
# for key in need_dic.keys():
#     if need_dic[key] != []:
#         continue
#     if '男女' in key:
#         first_choose = boys_name_list[random.randint(0,len(boys_name_list)-1)]
#         while first_choose in keep_choose:
#             first_choose = boys_name_list[random.randint(0, len(boys_name_list) - 1)]
#         second_choose = girls_name_list[random.randint(0,len(girls_name_list)-1)]
#         while second_choose in keep_choose:
#             second_choose = girls_name_list[random.randint(0, len(girls_name_list) - 1)]
#         need_dic[key] = [first_choose,second_choose]
#         keep_choose=keep_choose+need_dic[key]
#     else:
#         all_list = boys_name_list + girls_name_list
#         random.shuffle(all_list)
#         this_choose = all_list[random.randint(0,len(all_list)-1)]
#         while this_choose in keep_choose:
#             this_choose = all_list[random.randint(0, len(all_list) - 1)]
#         need_dic[key] = [this_choose]
#         keep_choose = keep_choose + need_dic[key]
# print(need_dic)
