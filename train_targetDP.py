import torch
import torch.nn as nn
import numpy as np
import random

from opacus import PrivacyEngine
from opacus.utils import module_modification
from sklearn.metrics import accuracy_score
from sklearn import datasets

from torchvision import datasets

from data_partition import clipDataTopX, shuffleAndSplitData
from dataset import readMNISTBin, readcifar10_GAN, readCIFAR10, readCIFAR100, readMINST, readNews, readFLW, readLocation_train, \
    readLocation_test, \
    readPurchase10_test, readPurchase10_train, readPurchase50_test, readPurchase50_train, readPurchase2_train, \
    readPurchase2_test, readPurchase20_train, readPurchase20_test, readPurchase100_test, readPurchase100_train

from net.CNN import CNN_Model
from net.NN import NN_Model
from net.alexnet_cifar10 import AlexNet
from net.CIFAR import cifar10_model
from net.cnn1 import cnn1
from net.cnn2 import BadNet
 
# from paint_distribution import paint_histogram
from preprocessing import preprocessingCIFAR, preprocessingCIFAR_GAN, preprocessingMINST, preprocessingNews, \
    preprocessingLFW
from train import CrossEntropy_L2, iterate_minibatches
from torch import optim
import torchvision.transforms as transforms
# from tensorboardX import SummaryWriter
import csv
import codecs
import pandas as pd
import json
import math


# def shuffleAndSplitData(dataX, dataY, cluster):
#
#     c = list(zip(dataX, dataY))
#     random.shuffle(c)
#     dataX, dataY = zip(*c)
#
#     data_x = np.array(dataX[:cluster])
#     data_y = np.array(dataY[:cluster])
#
#     return  data_x,data_y

def shuffleAndSplitData_news(dataX, dataY, cluster):
    c = list(zip(dataX, dataY))
    random.shuffle(c)
    dataX, dataY = zip(*c)

    data_x = np.array(dataX[:cluster])
    data_y = np.array(dataY[:cluster])
    test_x = np.array(dataX[cluster:cluster * 2])
    test_y = np.array(dataY[cluster:cluster * 2])

    return data_x, data_y, test_x, test_y


def readFashionMnist():
    data_path = 'dataset'
    train = datasets.FashionMNIST(data_path, train=True, download=True)
    test = datasets.FashionMNIST(data_path, train=False, download=False)
    x, y = train.data.numpy(), train.targets.numpy()
    xTest, yTest = test.data.numpy(), test.targets.numpy()
    x = x[:, np.newaxis, :, :]
    xTest = xTest[:, np.newaxis, :, :]
    return x, y, xTest, yTest


def preprocessingFashionMnist(toTrainData, toTestData):
    offset = np.mean(toTrainData, 0)
    scale = np.std(toTrainData, 0).clip(min=1)

    def rescale(raw_data):
        return (raw_data - offset) / scale

    return rescale(toTrainData), rescale(toTestData)


def train_traget_model(toTrainData,
                       toTrainLabel,
                       toTestData,
                       toTestLabel,
                       epochs=50,
                       batch_size=100,
                       n_hidden=128,
                       l2_ratio=1e-07,
                       learning_rate=0.02,
                       model_style='DP'
                       ):
    toTrainData = toTrainData.astype(np.float32)
    toTrainLabel = toTrainLabel.astype(np.int32)
    toTestData = toTestData.astype(np.float32)
    toTestLabel = toTestLabel.astype(np.int32)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    n_in = toTrainData.shape
    n_out = len(np.unique(toTrainLabel))

    if batch_size > len(toTrainLabel):
        batch_size = len(toTrainLabel)

    net = cifar10_model()
    # net = CNN_Model()
    # net = BadNet()
    # net = cnn1()
    # net = AlexNet()
    net.to(device)

    m = n_in[0]

    # criterion = CrossEntropy_L2(net, m, l2_ratio).to(device)
    if model_style == 'DP':
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.99)
        visual_batch_size = 1024
        visual_batch_rate = int(visual_batch_size / batch_size)
        privacy_engine = PrivacyEngine(
            net,
            batch_size=1024,  # batch_size=256    epoch=90  lr=0.001
            sample_size=len(toTrainLabel),
            alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
            noise_multiplier=0.91,  # 1.1
            max_grad_norm=0.6,  # 1.0      little=0.1
            # noise_multiplier= 0,
            # max_grad_norm = 10000,
            # secure_rng=False,
            # target_delta=1e-5,
            # C_0 = 1.2,
            # mu_0 = 0.6341586649417876,
            # T = 2343,
            # rho_c = 2,
            # rho_mu = 2
        )
        privacy_engine.attach(optimizer)
        # privacy_engine.attach(optimizer_ExpLR)
        # privacy_engine.attach(ExpLR)

        print('Training...')

        temp_loss = 0.0
        top1_acc = []
        epoch_acc = {}
        epoch_loss = {}
        gradNorm = {}
        accuracy = {}
        dynamic_c = {}
        dynamic_noise = {}

        # filename_dynamic_c = 'fashionmnist_result/dec/sigma_1.12/clipping_values/fashionmnist_lr_0.1_batch1024_sigma_1.12_relu_c1.2_garma_0.7_6.json'
        # filename_dynamic_noise = 'fashionmnist_result/dynamic/sigma_1.12/clipping_values/fashionmnist_lr_1_batch1024_sigma_1.12_relu_c1.2_2.json'
        # filename_epoch_acc = '/kolla/lgb/CIFAR10_mou/eps_3.0_mou_0.93_tanh.json'
        filename_epoch_acc = '/kolla/lgb/dynamic_momentum/cifar10/tanh/m_0.99_eps_7.56.json'
        # gradNorm = {}
        # accuracy = {}
        # C_begin =1.0
        # filename = 'imagenette/resnet/yuda/clipping_values/imagenette_lr0.001_momentum0.9_batch64_sigma_0.6_relu_5_6.json'
        # filename_ = 'imagenette/resnet/yuda/acc/imagenette_lr0.001_momentum0.9_batch64_sigma_0.6_relu_5_6.json'

        # 初始动量设置 ： momentum = 0.99
        initialMomentum = 0.55
        # mul = 0
        for epoch in range(epochs):
            losses = []
            gradNorm[epoch] = privacy_engine.max_grad_norm
            net.train()

            # yuda
            # privacy_engine.max_grad_norm = C_begin / min(2, 1 + epoch / epochs)
            # gradNorm[epoch] = privacy_engine.max_grad_norm
            # print(optimizer.param_groups[0]["momentum"])
            # # # # if(epoch!= 0 and epoch < 5):
            # # # #     mul += 1
            # initialMomentum -= 0.01
            # print("------------------------------------------")
            # optimizer.param_groups[0]["momentum"] = initialMomentum
            # print(optimizer.param_groups[0]["momentum"])
            # elif(epoch >= 5 and epoch % 5 == 0):
            #     mul += 1
            #     print("------------------------------------------")
            #     optimizer.param_groups[0]["momentum"] = initialMomentum * math.pow(scaleFactor , mul)

            dynamic_c[epoch] = []
            dynamic_noise[epoch] = []

            for i, (input_batch, target_batch) in enumerate(iterate_minibatches(toTrainData, toTrainLabel, batch_size)):

                dynamic_c[epoch].append(privacy_engine.max_grad_norm)
                dynamic_noise[epoch].append(privacy_engine.noise_multiplier)

                input_batch, target_batch = torch.tensor(input_batch).contiguous(), torch.tensor(target_batch).type(
                    torch.long)
                input_batch, target_batch = input_batch.to(device), target_batch.to(device)

                # empty parameters in optimizer
                # optimizer.zero_grad()

                outputs = net(input_batch)
                # outputs [100, 10]

                # calculate loss value
                loss = criterion(outputs, target_batch)
                preds = np.argmax(outputs.detach().cpu().numpy(), axis=1)
                acc1 = (preds == target_batch.detach().cpu().numpy()).mean()
                losses.append(loss.item())
                top1_acc.append(acc1)

                # back propagation
                loss.backward()

                # update paraeters in optimizer(update weight)
                if ((i + 1) % visual_batch_rate == 0) or ((i + 1) == int(len(toTrainLabel) / batch_size)):
                    optimizer.step()
                    # optimizer_ExpLR.step()
                else:
                    # optimizer_ExpLR.step()
                    # optimizer.step()
                    optimizer.virtual_step()

                temp_loss += loss.item()

                if (i + 1) % 49 == 0:
                    epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(1e-5)
                    print(
                        # f"\tLearning rate: {ExpLR.get_lr()} \t"
                        f"Train Epoch: {epoch} "
                        f"Loss: {loss.item():.6f} "
                        f"Acc@1: {np.mean(top1_acc):.6f} "
                        f"(ε = {epsilon:.4f}, δ = {1e-5}) for α = {best_alpha}"
                    )
            # ExpLR.step()
            # writer.add_scalars("DPLoss", {"Train": temp_loss}, epoch)
            print('Epoch {} Loss {}'.format(epoch, np.mean(losses)))
            epoch_loss[epoch] = np.mean(losses)
            temp_loss = 0.0

            net.eval()
            pred_y = []
            with torch.no_grad():
                for input_batch, _ in iterate_minibatches(toTestData, toTestLabel, batch_size, shuffle=False):
                    input_batch = torch.tensor(input_batch).contiguous()
                    input_batch = input_batch.to(device)

                    outputs = net(input_batch)

                    pred_y.append(torch.max(outputs, 1)[1].data.cpu().numpy())

                pred_y = np.concatenate(pred_y)

            print('Test Accuracy: {}'.format(accuracy_score(toTestLabel, pred_y)))
            accuracy[epoch] = accuracy_score(toTestLabel, pred_y)
            epoch_acc[epoch] = accuracy_score(toTestLabel, pred_y)

        print(epoch_acc)

        # with open(filename_dynamic_c, 'w') as json_file:
        # json.dump(dynamic_c, json_file, indent=4)
        # with open(filename_dynamic_noise, 'w') as json_file:
        #     json.dump(dynamic_noise, json_file, indent=4)
        with open(filename_epoch_acc, 'w') as json_file:
            json.dump(epoch_acc, json_file, indent=4)
        # np.save('.//badnet/sigma_3.45/dec_1.5_5.npy', epoch_acc)

    else:

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.99)
        temp_loss = 0.0
        iteration = 0
        epoch_acc = {}
        accuracy = {}
        print('Training...')
        net.train()
        normDict = {}
        filename_epoch_acc = '/kolla/lgb/dynamic_momentum/fashionmnist/no_private/m_0.99.json'
        for epoch in range(epochs):
            losses = []
            for input_batch, target_batch in iterate_minibatches(toTrainData, toTrainLabel, batch_size):
                input_batch, target_batch = torch.tensor(input_batch), torch.tensor(target_batch).type(torch.long)
                input_batch, target_batch = input_batch.to(device), target_batch.to(device)

                optimizer.zero_grad()
                outputs = net(input_batch)
                loss = criterion(outputs, target_batch)
                loss.backward()
                optimizer.step()

                # for n, p in net.named_parameters():
                #     print(n, p.grad_sample)

                # normLayer = torch.norm(net.state_dict()[name], p=2)
                # if name not in normDict:
                #     normDict[name] = []
                # normDict[name].append(np.array(normLayer.cpu()))

                losses.append(loss.item())

            # temp_loss = round(temp_loss, 3)

            # if epoch % 5 == 0:
            print('Epoch {}, train loss {}'.format(epoch, np.mean(losses)))

            # writer.add_scalars("Loss", {"Train": temp_loss}, epoch)

            temp_loss = 0.0

            net.eval()
            pred_y = []
            with torch.no_grad():
                for input_batch, _ in iterate_minibatches(toTestData, toTestLabel, batch_size, shuffle=False):
                    input_batch = torch.tensor(input_batch).contiguous()
                    input_batch = input_batch.to(device)

                    outputs = net(input_batch)

                    pred_y.append(torch.max(outputs, 1)[1].data.cpu().numpy())

                pred_y = np.concatenate(pred_y)

            print('Test Accuracy: {}'.format(accuracy_score(toTestLabel, pred_y)))
            accuracy[epoch] = accuracy_score(toTestLabel, pred_y)
            epoch_acc[epoch] = accuracy_score(toTestLabel, pred_y)

        print(epoch_acc)

        # with open(filename_dynamic_c, 'w') as json_file:
        # json.dump(dynamic_c, json_file, indent=4)
        # with open(filename_dynamic_noise, 'w') as json_file:
        #     json.dump(dynamic_noise, json_file, indent=4)
        with open(filename_epoch_acc, 'w') as json_file:
            json.dump(epoch_acc, json_file, indent=4)
        # np.save('.//badnet/sigma_3.45/dec_1.5_5.npy', epoch_acc)


def shuffleData(datax, datay):
    c = list(zip(datax, datay))
    random.shuffle(c)
    dataX, dataY = zip(*c)
    dataX = np.array(dataX)
    dataY = np.array(dataY)

    return dataX, dataY


if __name__ == '__main__':

    targetx, targety, target_testx, target_testy = readCIFAR10()
    targetx, targety = shuffleData(targetx, targety)
    target_testx, target_testy = shuffleData(target_testx, target_testy)
    targetx, target_testx = preprocessingCIFAR(targetx, target_testx)

    print(targetx.shape)
    print(target_testx.shape)

    train_traget_model(toTrainData=targetx,
                       toTrainLabel=targety,
                       toTestData=target_testx,
                       toTestLabel=target_testy,
                       epochs=40,
                       batch_size=1024,
                       n_hidden=128,
                       l2_ratio=1e-07,
                       learning_rate=1,
                       model_style='DP',
                       )
