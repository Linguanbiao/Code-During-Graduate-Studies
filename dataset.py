# ！-*- coding:utf-8 -*-

import pickle
import random

import numpy as np
import torch
import torchvision
from numpy import moveaxis
from sklearn import model_selection, datasets
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def readCIFAR10():
    data_path = './Dataset_cifar/cifar-10-batches-py'
    for i in range(5):
        f = open(data_path + '/data_batch_' + str(i + 1), 'rb')

        train_data_dict = pickle.load(f, encoding='iso-8859-1')

        f.close()
        if i == 0:
            X = train_data_dict["data"]
            y = train_data_dict["labels"]
            continue

        X = np.concatenate((X, train_data_dict["data"]), axis=0)
        y = np.concatenate((y, train_data_dict["labels"]), axis=0)

    f = open(data_path + '/test_batch', 'rb')
    test_data_dict = pickle.load(f, encoding='iso-8859-1')
    f.close()

    XTest = np.array(test_data_dict["data"])
    yTest = np.array(test_data_dict["labels"])
    return X, y, XTest, yTest


def readCIFAR100():
    f = open('/kolla/lgb/dataset/cifar-100-python/train', 'rb')
    train_data_dict = pickle.load(f, encoding='iso-8859-1')
    f.close()

    X = train_data_dict['data']
    y = train_data_dict['fine_labels']

    f = open('/kolla/lgb/dataset/cifar-100-python/test', 'rb')
    test_data_dict = pickle.load(f, encoding='iso-8859-1')
    f.close()

    XTest = np.array(test_data_dict['data'])
    yTest = np.array(test_data_dict['fine_labels'])

    return X, y, XTest, yTest


def readPurchase100():
    # print("use max_label")
    # data_path = 'data/purchase/max_label.csv'
    data_path = 'data/purchase/purchase100/purchase100_label.csv'
    f = pd.read_csv(data_path, header=None, skiprows=1)

    # normalize the values
    x_range = [i for i in range(600)]
    trainX = f[x_range].values
    trainY = f[600].values

    print(trainX.shape)
    print(trainY.shape)
    print(trainY)

    trainX = np.array(trainX)
    trainY = np.array(trainY)

    return trainX, trainY


def readPurchase50():
    # print("use max_label")
    # data_path = 'data/purchase/max_label.csv'
    data_path = './data/purchase/purchase50/purchase50_label.csv'
    f = pd.read_csv(data_path, header=None, skiprows=1)

    # normalize the values
    x_range = [i for i in range(600)]
    trainX = f[x_range].values
    trainY = f[600].values

    print(trainX.shape)
    print(trainY.shape)
    print(trainY)

    trainX = np.array(trainX)
    trainY = np.array(trainY)

    return trainX, trainY


def readPurchase50_train():
    # print("use max_label")
    # data_path = 'data/purchase/max_label.csv'
    data_path = './data/purchase/purchase50/max_label.csv'
    f = pd.read_csv(data_path, header=None, skiprows=1, nrows=10000)

    # normalize the values
    x_range = [i for i in range(600)]
    ori_targettrainX = f[x_range].values
    ori_targettrainY = f[600].values

    print(ori_targettrainX.shape)
    print(ori_targettrainY.shape)

    data_path = './data/purchase/purchase50/max_label.csv'
    f = pd.read_csv(data_path, header=None, skiprows=10005, nrows=10000)

    # normalize the values
    x_range = [i for i in range(600)]
    ori_shadowtrainX = f[x_range].values
    ori_shadowtrainY = f[600].values

    print(ori_shadowtrainX.shape)
    print(ori_shadowtrainY.shape)
    ori_targettrainX = np.array(ori_targettrainX)
    ori_targettrainY = np.array(ori_targettrainY)
    ori_shadowtrainX = np.array(ori_shadowtrainX)
    ori_shadowtrainY = np.array(ori_shadowtrainY)

    return ori_targettrainX, ori_targettrainY, ori_shadowtrainX, ori_shadowtrainY


def readPurchase50_test():
    # print("use max_label")
    # data_path = 'data/purchase/max_label.csv'
    data_path = './data/purchase/purchase50/max_label.csv'
    f = pd.read_csv(data_path, header=None, skiprows=10005, nrows=10000)

    # normalize the values
    x_range = [i for i in range(600)]
    target_testx = f[x_range].values
    target_testy = f[600].values

    print(target_testx.shape)
    print(target_testy.shape)

    data_path = './data/purchase/purchase50/max_label.csv'
    f = pd.read_csv(data_path, header=None, skiprows=30015, nrows=10000)

    # normalize the values
    x_range = [i for i in range(600)]
    shadow_testx = f[x_range].values
    shadow_testy = f[600].values
    print(shadow_testx.shape)
    print(shadow_testy.shape)

    target_testx = np.array(target_testx)
    target_testy = np.array(target_testy)
    shadow_testx = np.array(shadow_testx)
    shadow_testy = np.array(shadow_testy)

    return target_testx, target_testy, shadow_testx, shadow_testy


def readPurchase50_CTGAN_shadow():
    # print("use max_label")
    # data_path = 'data/purchase/max_label.csv'
    data_path = './data/purchase/purchase50/GAN_custom_kmeans_shadowmodel_max_10005.csv'
    f = pd.read_csv(data_path, header=None, skiprows=1, nrows=30000)

    # normalize the values
    x_range = [i for i in range(600)]
    trainX = f[x_range].values
    trainY = f[600].values

    print(trainX.shape)
    print(trainY.shape)
    print(trainY)

    trainX = np.array(trainX)
    trainY = np.array(trainY)

    return trainX, trainY


def readPurchase50_CTGAN_target():
    # print("use max_label")
    # data_path = 'data/purchase/max_label.csv'
    data_path = './data/purchase/purchase50/GAN_custom_kmeans_targetmodel_max_10005 .csv'
    f = pd.read_csv(data_path, header=None, skiprows=1, nrows=30000)

    # normalize the values
    x_range = [i for i in range(600)]
    trainX = f[x_range].values
    trainY = f[600].values

    print(trainX.shape)
    print(trainY.shape)
    print(trainY)
    trainX = np.array(trainX)
    trainY = np.array(trainY)
    return trainX, trainY


###########################################  Purchase100 ##############################
def readPurchase100_CTGAN_target():
    # print("use max_label")
    # data_path = 'data/purchase/max_label.csv'
    data_path = '../data/purchase/purchase100/purchase100_targetmodel_gan.csv'
    f = pd.read_csv(data_path, header=None, skiprows=1, nrows=22000)

    # normalize the values
    x_range = [i for i in range(600)]
    trainX = f[x_range].values
    trainY = f[600].values

    print(trainX.shape)
    print(trainY.shape)
    print(trainY)
    trainX = np.array(trainX)
    trainY = np.array(trainY)
    return trainX, trainY


def readPurchase100_CTGAN_shadow():
    # print("use max_label")
    # data_path = 'data/purchase/max_label.csv'
    data_path = 'data/purchase/purchase100/purchase100_shadowmodel_gan.csv'
    f = pd.read_csv(data_path, header=None, skiprows=1, nrows=50000)

    # normalize the values
    x_range = [i for i in range(600)]
    trainX = f[x_range].values
    trainY = f[600].values

    print(trainX.shape)
    print(trainY.shape)
    print(trainY)
    trainX = np.array(trainX)
    trainY = np.array(trainY)
    return trainX, trainY


def readPurchase100_test():
    # print("use max_label")
    # data_path = 'data/purchase/max_label.csv'
    data_path = '../data/purchase/purchase100/df_remaining_purchase100.csv'
    f = pd.read_csv(data_path, header=None, skiprows=100, nrows=10005)

    # normalize the values
    x_range = [i for i in range(600)]
    target_testx = f[x_range].values
    target_testy = f[600].values

    print(target_testx.shape)
    print(target_testy.shape)

    data_path = '../data/purchase/purchase100/df_remaining_purchase100.csv'
    f = pd.read_csv(data_path, header=None, skiprows=25000, nrows=10005)

    # normalize the values
    x_range = [i for i in range(600)]
    shadow_testx = f[x_range].values
    shadow_testy = f[600].values
    print(shadow_testx.shape)
    print(shadow_testy.shape)

    target_testx = np.array(target_testx)
    target_testy = np.array(target_testy)
    shadow_testx = np.array(shadow_testx)
    shadow_testy = np.array(shadow_testy)

    return target_testx, target_testy, shadow_testx, shadow_testy


def readPurchase100_train():
    # print("use max_label")
    # data_path = 'data/purchase/max_label.csv'
    data_path = '../data/purchase/purchase100/df_target_totrain.csv'
    f = pd.read_csv(data_path, header=None, skiprows=1)

    # normalize the values
    x_range = [i for i in range(600)]
    ori_targettrainX = f[x_range].values
    ori_targettrainY = f[600].values

    print(ori_targettrainX.shape)
    print(ori_targettrainY.shape)

    data_path = '../data/purchase/purchase100/df_shadow_totrain.csv'
    f = pd.read_csv(data_path, header=None, skiprows=1)

    # normalize the values
    x_range = [i for i in range(600)]
    ori_shadowtrainX = f[x_range].values
    ori_shadowtrainY = f[600].values

    print(ori_shadowtrainX.shape)
    print(ori_shadowtrainY.shape)
    ori_targettrainX = np.array(ori_targettrainX)
    ori_targettrainY = np.array(ori_targettrainY)
    ori_shadowtrainX = np.array(ori_shadowtrainX)
    ori_shadowtrainY = np.array(ori_shadowtrainY)

    return ori_targettrainX, ori_targettrainY, ori_shadowtrainX, ori_shadowtrainY

###################################### Purchase100 ##########################################

###################################### Purchase10  ###################################


def readPurchase10():
    # print("use max_label")
    # data_path = 'data/purchase/max_label.csv'
    data_path = 'data/purchase/purchase10/purchase10.csv'
    f = pd.read_csv(data_path, header=None, skiprows=1)

    # normalize the values
    x_range = [i for i in range(600)]
    trainX = f[x_range].values
    trainY = f[600].values

    print(trainX.shape)
    print(trainY.shape)
    print(trainY)

    trainX = np.array(trainX)
    trainY = np.array(trainY)

    return trainX, trainY


def readPurchase10_CTGAN_target():
    # print("use max_label")
    # data_path = 'data/purchase/max_label.csv'
    data_path = '../data/purchase/purchase10/purchase10_targetmodel_gan.csv'
    f = pd.read_csv(data_path, header=None, skiprows=1, nrows=22000)

    # normalize the values
    x_range = [i for i in range(600)]
    trainX = f[x_range].values
    trainY = f[600].values

    print(trainX.shape)
    print(trainY.shape)
    print(trainY)
    trainX = np.array(trainX)
    trainY = np.array(trainY)
    return trainX, trainY


def readPurchase10_CTGAN_shadow():
    # print("use max_label")
    # data_path = 'data/purchase/max_label.csv'
    data_path = 'data/purchase/purchase10/purchase10_shadowmodel_gan.csv'
    f = pd.read_csv(data_path, header=None, skiprows=1, nrows=15000)

    # normalize the values
    x_range = [i for i in range(600)]
    trainX = f[x_range].values
    trainY = f[600].values

    print(trainX.shape)
    print(trainY.shape)
    print(trainY)
    trainX = np.array(trainX)
    trainY = np.array(trainY)
    print(tran)
    return trainX, trainY


def readPurchase10_train():
    # print("use max_label")
    # data_path = 'data/purchase/max_label.csv'
    data_path = './data/purchase/purchase10/df_target_totrain.csv'
    # data_path = '../data/purchase/purchase10/purchase10.csv'
    f = pd.read_csv(data_path, header=None, skiprows=1, nrows=10000)

    # normalize the values
    x_range = [i for i in range(600)]
    ori_targettrainX = f[x_range].values
    ori_targettrainY = f[600].values

    print(ori_targettrainX.shape)
    print(ori_targettrainY.shape)

    data_path = './data/purchase/purchase10/df_shadow_totrain.csv'
    f = pd.read_csv(data_path, header=None, skiprows=1, nrows=10000)

    # normalize the values
    x_range = [i for i in range(600)]
    ori_shadowtrainX = f[x_range].values
    ori_shadowtrainY = f[600].values

    print(ori_shadowtrainX.shape)
    print(ori_shadowtrainY.shape)
    ori_targettrainX = np.array(ori_targettrainX)
    ori_targettrainY = np.array(ori_targettrainY)
    ori_shadowtrainX = np.array(ori_shadowtrainX)
    ori_shadowtrainY = np.array(ori_shadowtrainY)

    return ori_targettrainX, ori_targettrainY, ori_shadowtrainX, ori_shadowtrainY


def readPurchase10_test():
    # print("use max_label")
    # data_path = 'data/purchase/max_label.csv'
    data_path = './data/purchase/purchase10/df_remaining_purchase10.csv'
    # data_path = '../data/purchase/purchase10/purchase10.csv'
    f = pd.read_csv(data_path, header=None, skiprows=100, nrows=10000)

    # normalize the values
    x_range = [i for i in range(600)]
    target_testx = f[x_range].values
    target_testy = f[600].values

    print(target_testx.shape)
    print(target_testy.shape)

    data_path = './data/purchase/purchase10/df_remaining_purchase10.csv'
    f = pd.read_csv(data_path, header=None, skiprows=30000, nrows=10000)

    # normalize the values
    x_range = [i for i in range(600)]
    shadow_testx = f[x_range].values
    shadow_testy = f[600].values
    print(shadow_testx.shape)
    print(shadow_testy.shape)

    target_testx = np.array(target_testx)
    target_testy = np.array(target_testy)
    shadow_testx = np.array(shadow_testx)
    shadow_testy = np.array(shadow_testy)

    return target_testx, target_testy, shadow_testx, shadow_testy
# ____________________________________Purchase2______________________________________


def readPurchase2():
    # print("use max_label")
    # data_path = 'data/purchase/max_label.csv'
    data_path = 'data/purchase/purchase2/purchase2.csv'
    f = pd.read_csv(data_path, header=None, skiprows=1)

    # normalize the values
    x_range = [i for i in range(600)]
    trainX = f[x_range].values
    trainY = f[600].values

    print(trainX.shape)
    print(trainY.shape)
    print(trainY)

    trainX = np.array(trainX)
    trainY = np.array(trainY)

    return trainX, trainY


def readPurchase2_CTGAN_target():
    # print("use max_label")
    # data_path = 'data/purchase/max_label.csv'
    data_path = '../data/purchase/purchase2/purchase2_targetmodel_gan.csv'
    f = pd.read_csv(data_path, header=None, skiprows=1, nrows=15000)

    # normalize the values
    x_range = [i for i in range(600)]
    trainX = f[x_range].values
    trainY = f[600].values

    print(trainX.shape)
    print(trainY.shape)
    print(trainY)
    trainX = np.array(trainX)
    trainY = np.array(trainY)
    return trainX, trainY


def readPurchase2_CTGAN_shadow():
    # print("use max_label")
    # data_path = 'data/purchase/max_label.csv'
    data_path = './data/purchase/purchase2/purchase2_shadowmodel_gan.csv'
    f = pd.read_csv(data_path, header=None, skiprows=1, nrows=15000)

    # normalize the values
    x_range = [i for i in range(600)]
    trainX = f[x_range].values
    trainY = f[600].values

    print(trainX.shape)
    print(trainY.shape)
    print(trainY)
    trainX = np.array(trainX)
    trainY = np.array(trainY)
    return trainX, trainY


def readPurchase2_train():
    # print("use max_label")
    # data_path = 'data/purchase/max_label.csv'
    data_path = '../data/purchase/purchase2/df_target_totrain.csv'
    # data_path = '../data/purchase/purchase10/purchase10.csv'
    f = pd.read_csv(data_path, header=None, skiprows=1)

    # normalize the values
    x_range = [i for i in range(600)]
    ori_targettrainX = f[x_range].values
    ori_targettrainY = f[600].values

    print(ori_targettrainX.shape)
    print(ori_targettrainY.shape)

    data_path = '../data/purchase/purchase2/df_shadow_totrain.csv'
    f = pd.read_csv(data_path, header=None, skiprows=1)

    # normalize the values
    x_range = [i for i in range(600)]
    ori_shadowtrainX = f[x_range].values
    ori_shadowtrainY = f[600].values

    print(ori_shadowtrainX.shape)
    print(ori_shadowtrainY.shape)
    ori_targettrainX = np.array(ori_targettrainX)
    ori_targettrainY = np.array(ori_targettrainY)
    ori_shadowtrainX = np.array(ori_shadowtrainX)
    ori_shadowtrainY = np.array(ori_shadowtrainY)

    return ori_targettrainX, ori_targettrainY, ori_shadowtrainX, ori_shadowtrainY


def readPurchase2_test():
    # print("use max_label")
    # data_path = 'data/purchase/max_label.csv'
    data_path = '../data/purchase/purchase2/df_remaining_purchase2.csv'
    # data_path = '../data/purchase/purchase10/purchase10.csv'
    f = pd.read_csv(data_path, header=None, skiprows=100, nrows=10005)
    #
    # normalize the values
    x_range = [i for i in range(600)]
    target_testx = f[x_range].values
    target_testy = f[600].values

    print(target_testx.shape)
    print(target_testy.shape)
    #
    data_path = '../data/purchase/purchase2/df_remaining_purchase2.csv'
    f = pd.read_csv(data_path, header=None, skiprows=30000, nrows=10005)

    # normalize the values
    x_range = [i for i in range(600)]
    shadow_testx = f[x_range].values
    shadow_testy = f[600].values
    print(shadow_testx.shape)
    print(shadow_testy.shape)

    target_testx = np.array(target_testx)
    target_testy = np.array(target_testy)
    shadow_testx = np.array(shadow_testx)
    shadow_testy = np.array(shadow_testy)

    return target_testx, target_testy, shadow_testx, shadow_testy
    # return target_testx,target_testy
# _____________________________________  Purchase2 __________________________________
######################################  Purchase10 #################################
# ______________________________________ Purchase20 ————————————————————————————————


def readPurchase20():
    # print("use max_label")
    # data_path = 'data/purchase/max_label.csv'
    data_path = 'data/purchase/purchase20/purchase20.csv'
    f = pd.read_csv(data_path, header=None, skiprows=1)

    # normalize the values
    x_range = [i for i in range(600)]
    trainX = f[x_range].values
    trainY = f[600].values

    print(trainX.shape)
    print(trainY.shape)
    print(trainY)

    trainX = np.array(trainX)
    trainY = np.array(trainY)

    return trainX, trainY


def readPurchase20_CTGAN_target():
    # print("use max_label")
    # data_path = 'data/purchase/max_label.csv'
    data_path = '../data/purchase/purchase20/purchase20_targetmodel_gan.csv'
    f = pd.read_csv(data_path, header=None, skiprows=1, nrows=15000)

    # normalize the values
    x_range = [i for i in range(600)]
    trainX = f[x_range].values
    trainY = f[600].values

    print(trainX.shape)
    print(trainY.shape)
    print(trainY)
    trainX = np.array(trainX)
    trainY = np.array(trainY)
    return trainX, trainY


def readPurchase20_CTGAN_shadow():
    # print("use max_label")
    # data_path = 'data/purchase/max_label.csv'
    data_path = 'data/purchase/purchase20/purchase20_shadowmodel_gan.csv'
    f = pd.read_csv(data_path, header=None, skiprows=1, nrows=20000)

    # normalize the values
    x_range = [i for i in range(600)]
    trainX = f[x_range].values
    trainY = f[600].values

    print(trainX.shape)
    print(trainY.shape)
    print(trainY)
    trainX = np.array(trainX)
    trainY = np.array(trainY)
    return trainX, trainY


def readPurchase20_train():
    # print("use max_label")
    # data_path = 'data/purchase/max_label.csv'
    data_path = '../data/purchase/purchase20/df_target_totrain.csv'
    # data_path = '../data/purchase/purchase20/purchase20.csv'
    f = pd.read_csv(data_path, header=None, skiprows=1)

    # normalize the values
    x_range = [i for i in range(600)]
    ori_targettrainX = f[x_range].values
    ori_targettrainY = f[600].values

    print(ori_targettrainX.shape)
    print(ori_targettrainY.shape)

    data_path = '../data/purchase/purchase20/df_shadow_totrain.csv'
    # data_path = '../data/purchase/purchase20/purchase20.csv'
    f = pd.read_csv(data_path, header=None, skiprows=1)

    # normalize the values
    x_range = [i for i in range(600)]
    ori_shadowtrainX = f[x_range].values
    ori_shadowtrainY = f[600].values

    print(ori_shadowtrainX.shape)
    print(ori_shadowtrainY.shape)
    ori_targettrainX = np.array(ori_targettrainX)
    ori_targettrainY = np.array(ori_targettrainY)
    ori_shadowtrainX = np.array(ori_shadowtrainX)
    ori_shadowtrainY = np.array(ori_shadowtrainY)

    return ori_targettrainX, ori_targettrainY, ori_shadowtrainX, ori_shadowtrainY


def readPurchase20_test():
    # print("use max_label")
    # data_path = 'data/purchase/max_label.csv'
    data_path = '../data/purchase/purchase20/df_remaining_purchase20.csv'
    f = pd.read_csv(data_path, header=None, skiprows=100, nrows=10005)

    # normalize the values
    x_range = [i for i in range(600)]
    target_testx = f[x_range].values
    target_testy = f[600].values

    print(target_testx.shape)
    print(target_testy.shape)

    data_path = '../data/purchase/purchase20/df_remaining_purchase20.csv'
    f = pd.read_csv(data_path, header=None, skiprows=30000, nrows=10005)

    # normalize the values
    x_range = [i for i in range(600)]
    shadow_testx = f[x_range].values
    shadow_testy = f[600].values
    print(shadow_testx.shape)
    print(shadow_testy.shape)

    target_testx = np.array(target_testx)
    target_testy = np.array(target_testy)
    shadow_testx = np.array(shadow_testx)
    shadow_testy = np.array(shadow_testy)

    return target_testx, target_testy, shadow_testx, shadow_testy
# ——————————————————————————————————————- Purchase20 ——————————————————————————————————

###################################### Location ######################################


def readLocation_CTGAN_shadow():
    # print("use max_label")
    # data_path = 'data/purchase/max_label.csv'
    data_path = 'data/location/location_shadowmodel_GAN.csv'
    f = pd.read_csv(data_path, header=None, skiprows=1, nrows=20000)

    # normalize the values
    x_range = [i for i in range(446)]
    trainX = f[x_range].values
    trainY = f[446].values

    print(trainX.shape)
    print(trainY.shape)
    print(trainY)
    trainX = np.array(trainX)
    trainY = np.array(trainY)
    return trainX, trainY


def readLocation_CTGAN_target():
    # print("use max_label")
    # data_path = 'data/purchase/max_label.csv'
    data_path = '../data/location/location_targetmodel_GAN.csv'
    f = pd.read_csv(data_path, header=None, skiprows=1, nrows=22000)

    # normalize the values
    x_range = [i for i in range(446)]
    trainX = f[x_range].values
    trainY = f[446].values

    print(trainX.shape)
    print(trainY.shape)
    print(trainY)
    trainX = np.array(trainX)
    trainY = np.array(trainY)
    return trainX, trainY


def readLocation_train():
    # print("use max_label")
    # data_path = 'data/purchase/max_label.csv'
    data_path = '../data/location/df_target_totrain.csv'
    f = pd.read_csv(data_path, header=None, skiprows=1)

    # normalize the values
    x_range = [i for i in range(446)]
    ori_targettrainX = f[x_range].values
    ori_targettrainY = f[446].values

    print(ori_targettrainX.shape)
    print(ori_targettrainY.shape)

    data_path = '../data/location/df_shadow_totrain.csv'
    f = pd.read_csv(data_path, header=None, skiprows=1)

    # normalize the values
    x_range = [i for i in range(446)]
    ori_shadowtrainX = f[x_range].values
    ori_shadowtrainY = f[446].values

    print(ori_shadowtrainX.shape)
    print(ori_shadowtrainY.shape)
    ori_targettrainX = np.array(ori_targettrainX)
    ori_targettrainY = np.array(ori_targettrainY)
    ori_shadowtrainX = np.array(ori_shadowtrainX)
    ori_shadowtrainY = np.array(ori_shadowtrainY)

    return ori_targettrainX, ori_targettrainY, ori_shadowtrainX, ori_shadowtrainY


def readLocation_test():
    # print("use max_label")
    # data_path = 'data/purchase/max_label.csv'
    data_path = '../data/location/df_remaining_location.csv'
    f = pd.read_csv(data_path, header=None, skiprows=1, nrows=1250)

    # normalize the values
    x_range = [i for i in range(446)]
    target_testx = f[x_range].values
    target_testy = f[446].values

    print(target_testx.shape)
    print(target_testy.shape)

    data_path = '../data/location/df_remaining_location.csv'
    f = pd.read_csv(data_path, header=None, skiprows=1250, nrows=1250)

    # normalize the values
    x_range = [i for i in range(446)]
    shadow_testx = f[x_range].values
    shadow_testy = f[446].values
    print(shadow_testx.shape)
    print(shadow_testy.shape)

    target_testx = np.array(target_testx)
    target_testy = np.array(target_testy)
    shadow_testx = np.array(shadow_testx)
    shadow_testy = np.array(shadow_testy)

    return target_testx, target_testy, shadow_testx, shadow_testy
    # return target_testx,target_testy
######################################## Location #########################


def readFLW(data_path):
    data_path = data_path + '/lfw_funneled'
    lfw_people = datasets.fetch_lfw_people(data_home=data_path, min_faces_per_person=40, resize=1)

    n_samples, h, w = lfw_people.images.shape  # resize=0.4 (1867,50,37)  # resize=1.0 (1867, 125, 94)

    x = lfw_people.images.reshape(n_samples, 1, h, w) / 255.0
    y = lfw_people.target

    trainX, testX, trainY, testY = model_selection.train_test_split(x, y, test_size=.1, random_state=42)

    return trainX, trainY, testX, testY  # trainX(1680,1,50,37),testX(187,1,50,37)


def readMINST():
    data_path = './data/MINST/processed'
    train_data_dict = torch.load(data_path + '/training.pt')
    X, y = train_data_dict[0].numpy(), train_data_dict[1].numpy()

    for index in range(len(X)):
        X[index] = X[index].transpose((1, 0))

    X = X.reshape(X.shape[0], -1)

    test_data_dict = torch.load(data_path + '/test.pt')
    XTest, yTest = test_data_dict[0].numpy(), test_data_dict[1].numpy()

    for index in range(len(XTest)):
        XTest[index] = XTest[index].transpose((1, 0))

    XTest = XTest.reshape(XTest.shape[0], -1)

    return X, y, XTest, yTest


def readMNISTBin():
    file = np.load('./data/MNIST_Bin/imagesTrain_MNIST.npz', allow_pickle=True)
    imagesTrain, labelsTrain = file['images'].astype(np.float32)[:, np.newaxis], file['labels']
    file = np.load('./data/MNIST_Bin/imagesTest_MNIST.npz', allow_pickle=True)
    imagesTest, labelTest = file['images'].astype(np.float32)[:, np.newaxis], file['labels']
    return imagesTrain, labelsTrain, imagesTest, labelTest


def readAdult(data_path):
    data_path = data_path + '/adult.data'
    f = pd.read_csv(data_path, header=None)

    # encode the categorical values
    for col in [1, 3, 5, 6, 7, 8, 9, 13, 14]:
        le = LabelEncoder()
        f[col] = le.fit_transform(f[col].astype('str'))

    # normalize the values
    x_range = [i for i in range(14)]

    x = f[x_range].values
    y = f[14].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

    return x_train, y_train, x_test, y_test


def readNews(data_path):
    train = fetch_20newsgroups(data_home=data_path,
                               subset='train',
                               remove=('headers', 'footers', 'quotes'))

    test = fetch_20newsgroups(data_home=data_path,
                              subset='test',
                              remove=('headers', 'footers', 'quotes'))

    X = np.concatenate((train.data, test.data), axis=0)

    y = np.concatenate((train.target, test.target), axis=0)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(X)
    X = X.toarray()

    return X, y


def readcifar10_GAN(data_path):
    dataX = []
    dataY = []
    for i in range(10):
        datapath_toload = data_path + '/chosen_label_' + str(i) + '.npy'
        f = np.load(datapath_toload, allow_pickle=True).item()

        data_x = f['images']
        data_y = f['labels']

        dataX.append(data_x)
        dataY.append(data_y)

    dataX = np.concatenate(dataX, axis=0)
    dataY = np.concatenate(dataY, axis=0)

    datax = np.array(dataX, dtype=object)
    datay = np.array(dataY, dtype=object)

    return datax, datay


# adv_1

def readcifar10_shadow_train():
    data = np.load('./data/cifar10_adv1/shadow_train.npz')
    features = data['x']
    labels = data['y']

    features = np.array(features)
    labels = np.array(labels)

    return features, labels


def readcifar10_target_train():
    data = np.load('./data/cifar10_adv1/target_train.npz')
    features = data['x']
    labels = data['y']

    features = np.array(features)
    labels = np.array(labels)

    return features, labels


def readcifar10_shadow_test():
    data = np.load('./data/cifar10_adv1/shadow_test.npz')
    features = data['x']
    labels = data['y']

    features = np.array(features)
    labels = np.array(labels)

    return features, labels


def readcifar10_target_test():
    data = np.load('./data/cifar10_adv1/target_test.npz')
    features = data['x']
    labels = data['y']

    features = np.array(features)
    labels = np.array(labels)

    return features, labels


def readcifar10_StyleGAN_shadow():
    testx_, testy_ = [], []

    for label in range(10):
        data = np.load('./data/cifar10_adv1/GAN/shadow_sample_' + str(label) + '.npz', allow_pickle=True)
        testx = (moveaxis(data['images'], 3, 1))
        testy = (data['labels'])
        testx_.extend(np.array(testx, dtype=object)[:1000])
        testy_.extend(np.array(testy, dtype=object)[:1000])

    testx_ = np.array(testx_)
    testy_ = np.array(testy_)

    return testx_, testy_


def readcifar10_StyleGAN_target():
    testx_, testy_ = [], []

    for label in range(10):
        data = np.load('./data/cifar10_adv1/GAN/target_sample_' + str(label) + '.npz', allow_pickle=True)
        testx = (moveaxis(data['images'], 3, 1))
        testy = (data['labels'])
        testx_.extend(np.array(testx, dtype=object)[:1500])
        testy_.extend(np.array(testy, dtype=object)[:1500])

    testx_ = np.array(testx_)
    testy_ = np.array(testy_)

    return testx_, testy_


# adv_1

def readcifar10_StyleGAN_test():
    testx_, testy_ = [], []

    for label in range(10):
        data = np.load('../data/cifar10_GAN_FID10/sample_' + str(label) + '.npz', allow_pickle=True)
        testx = (moveaxis(data['images'], 3, 1))
        testy = (data['labels'])
        testx_.extend(np.array(testx, dtype=object)[1000:2000])
        testy_.extend(np.array(testy, dtype=object)[1000:2000])

    testx_ = np.array(testx_)
    testy_ = np.array(testy_)

    return testx_, testy_


def readcifar10_StyleGAN_all():
    testx_, testy_ = [], []
    for label in range(10):
        data = np.load('../data/cifar10_GAN_FID10/sample_' + str(label) + '.npz', allow_pickle=True)
        testx_.extend(moveaxis(data['images'], 3, 1))
        testy_.extend(data['labels'])

    print('The number of images:', len(testx_))
    testx_ = np.array(testx_)
    testy_ = np.array(testy_)
    return testx_, testy_


def readAdult_train():
    data_path = '../data/Adult/adult.data'
    f = pd.read_csv(data_path, header=None, error_bad_lines=False)

    # encode the categorical values
    for col in [1, 3, 5, 6, 7, 8, 9, 13, 14]:
        le = LabelEncoder()
        f[col] = le.fit_transform(f[col].astype('str'))

    # normalize the values
    x_range = [i for i in range(14)]
    x_train = f[x_range].values
    y_train = f[14].values

    return x_train, y_train


def readAdult_test():
    data_path = '../data/Adult/adult.test'
    f = pd.read_csv(data_path, header=None, error_bad_lines=False)

    # encode the categorical values
    for col in [1, 3, 5, 6, 7, 8, 9, 13, 14]:
        le = LabelEncoder()
        f[col] = le.fit_transform(f[col].astype('str'))

    # normalize the values
    x_range = [i for i in range(14)]
    x_test = f[x_range].values
    y_test = f[14].values
    return x_test, y_test


def readAdult_fake():
    data_path = '../data/fake_Adult/Adult_fake.csv'
    f = pd.read_csv(data_path, header=None, error_bad_lines=False)

    # encode the categorical values
    for col in [1, 3, 5, 6, 7, 8, 9, 13, 14]:
        le = LabelEncoder()
        f[col] = le.fit_transform(f[col].astype('str'))

    # normalize the values
    x_range = [i for i in range(14)]
    fake_train = f[x_range].values
    fake_test = f[14].values

    return fake_train, fake_test


def readLocation():
    npzdata = np.load('data/location/data_complete.npz')
    x_data = npzdata['x'][:, :]
    y_data = npzdata['y'][:]
    y_data = y_data - 1.0
    print(x_data.shape)
    print(y_data.shape)

    return x_data, y_data


# ________________________MNIST______________________________
def readMINST_StyleGAN_target():
    testx_, testy_ = [], []

    for label in range(10):
        data = np.load('./data/fake_mnist/target_sample_' + str(label) + '.npz', allow_pickle=True)
        # testx = (moveaxis(data['images'], 3, 1))
        # testy = (data['labels'])
        # testx_.extend(np.array(testx, dtype=object)[:1052])
        # testy_.extend(np.array(testy, dtype=object)[:1052])
        testx_.extend(np.array(data['images'][:, np.newaxis], dtype=object))
        testy_.extend(np.array(data['labels'], dtype=object))

    testx_ = np.array(testx_)
    testy_ = np.array(testy_)
    print(testx_.shape)

    return testx_, testy_


def readMINST_StyleGAN_shadow():
    testx_, testy_ = [], []

    for label in range(10):
        data = np.load('./data/fake_mnist/shadow_sample_' + str(label) + '.npz', allow_pickle=True)
        # testx = (moveaxis(data['images'], 3, 1))
        # testy = (data['labels'])
        testx_.extend(np.array(data['images'][:, np.newaxis], dtype=object))
        testy_.extend(np.array(data['labels'], dtype=object))

    testx_ = np.array(testx_)
    testy_ = np.array(testy_)

    print(testx_.shape)

    return testx_, testy_


def readMINTS_StyleGAN_test():
    data = np.load('./data/fake_mnist/MNIST_remain.npz')
    features = data['data']
    labels = data['labels']
    print(features.shape)
    print(labels.shape)
    dataX = features
    dataY = labels
    # c = list(zip(dataX,dataY))
    # random.shuffle(c)
    return dataX, dataY


def readMINTS_StyleGAN_ori():
    data = np.load('./data/fake_mnist/MNIST_test_10520.npz')
    target_features = data['data']
    target_labels = data['labels']
    print(target_features.shape)
    print(target_labels.shape)

    data_shadow = np.load('./data/fake_mnist/MNIST_train_10520.npz')
    shadow_features = data_shadow['data']
    shadow_labels = data_shadow['labels']
    print(shadow_features.shape)
    print(shadow_labels.shape)

    return target_features, target_labels, shadow_features, shadow_labels
# ____________________________MNIST_____________________________________
# ____________________________CIFAR100___________________________________


def readCIFAR100_StyleGAN_target():
    testx_, testy_ = [], []

    for label in range(100):
        data = np.load('./data/fake_cifar100/CIFAR100_target/cifar100_target_sample_' +
                       str(label) + '.npz', allow_pickle=True)
        # testx = (moveaxis(data['images'], 3, 1))
        # testy = (data['labels'])
        # testx_.extend(np.array(testx, dtype=object)[:1052])
        # testy_.extend(np.array(testy, dtype=object)[:1052])
        testx_.extend(np.array(data['images'].transpose(0, 3, 1, 2), dtype=object))
        testy_.extend(np.array(data['labels'], dtype=object))

    testx_ = np.array(testx_, dtype=np.float32)
    testy_ = np.array(testy_)
    print(testx_.shape)

    return testx_, testy_


def readCIFAR100_StyleGAN_shadow():
    testx_, testy_ = [], []

    for label in range(100):
        data = np.load('./data/fake_cifar100/CIFAR100_shadow/cifar100_shadow_sample_' +
                       str(label) + '.npz', allow_pickle=True)
        # testx = (moveaxis(data['images'], 3, 1))
        # testy = (data['labels'])
        testx_.extend(np.array(data['images'].transpose(0, 3, 1, 2), dtype=object))
        testy_.extend(np.array(data['labels'], dtype=object))

    testx_ = np.array(testx_, dtype=np.float32)
    testy_ = np.array(testy_)

    print(testx_.shape)

    return testx_, testy_


def readCIFAR100_StyleGAN_test():
    data = np.load('./data/fake_cifar100/CIFAR100_remain.npz')
    features = data['data']
    labels = data['labels']
    print(features.shape)
    print(labels.shape)
    dataX = features
    dataY = labels
    # c = list(zip(dataX,dataY))
    # random.shuffle(c)
    return dataX, dataY


def readCIFAR100_StyleGAN_ori():
    data = np.load('./data/fake_cifar100/CIFAR100_test_10520.npz')
    target_features = data['data']
    target_labels = data['labels']
    print(target_features.shape)
    print(target_labels.shape)

    data_shadow = np.load('./data/fake_cifar100/CIFAR100_train_10520.npz')
    shadow_features = data_shadow['data']
    shadow_labels = data_shadow['labels']
    print(shadow_features.shape)
    print(shadow_labels.shape)

    return target_features, target_labels, shadow_features, shadow_labels
# if __name__ =='__main__':
#     readLocation()
