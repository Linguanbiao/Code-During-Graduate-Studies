import torch
from torchvision import datasets as dset
from torchvision import transforms as T
from torch.utils.data import DataLoader, RandomSampler
from torchvision.datasets import ImageFolder
from torch import nn
import torch.optim as optim

import numpy as np
import copy
import pickle
import random

from sklearn.metrics import accuracy_score
from opacus import PrivacyEngine
from net.CNN import CNN_Model
from sklearn import model_selection, datasets
from torchvision import datasets
# from net.cifar10_model import cifar10_model
# from cnn2 import BadNet
# from net.Alexnet import AlexNet
# from net.lenet import LeNet5
# from net.ResNet import ResNet
# from net.googlenet import GoogLeNet
# from net.cnn1 import cnn1
# from net.cnn2 import BadNet
from net.alexnet_img import AlexNet
# from model.smallnet import cnn2
import warnings
import json
import statistics
import math

warnings.filterwarnings("ignore")


def readImagenette():
    cifar_mean = (0.4914, 0.4822, 0.4465)
    cifar_stddev = (0.2023, 0.1994, 0.2010)
    transforms = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(mean=cifar_mean, std=cifar_stddev)])
    trainset = ImageFolder('imagenette2/train', transform=transforms)
    testset = ImageFolder('imagenette2/val', transform=transforms)
    return trainset, testset


def readCIFAR10():
    data_path = 'dataset/cifar-10-batches-py'
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


def readMINST():
    data_path = 'dataset/MNIST/processed'
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


def readFashionMnist():
    data_path = 'dataset'
    train = datasets.FashionMNIST(data_path, train=True, download=True)
    test = datasets.FashionMNIST(data_path, train=False, download=False)
    x, y = train.data.numpy(), train.targets.numpy()
    xTest, yTest = test.data.numpy(), test.targets.numpy()
    x = x[:, np.newaxis, :, :]
    xTest = xTest[:, np.newaxis, :, :]
    return x, y, xTest, yTest


def shuffleData(datax, datay):
    c = list(zip(datax, datay))
    random.shuffle(c)
    dataX, dataY = zip(*c)
    # if len(dataX) == 60000:
    #     dataX = np.array(dataX[:6000])
    #     dataY = np.array(dataY[:6000])
    # elif len(dataX) == 50000:
    #     dataX = np.array(dataX[:5000])
    #     dataY = np.array(dataY[:5000])
    # else:
    dataX = np.array(dataX)
    dataY = np.array(dataY)
    return dataX, dataY


def preprocessingCIFAR(toTrainData, toTestData):
    def reshape_for_save(raw_data):
        raw_data = np.dstack(
            (raw_data[:, :1024], raw_data[:, 1024:2048], raw_data[:, 2048:]))
        raw_data = raw_data.reshape(
            (raw_data.shape[0], 32, 32, 3)).transpose(0, 3, 1, 2)
        return raw_data.astype(np.float32)

    offset = np.mean(reshape_for_save(toTrainData), 0)
    scale = np.std(reshape_for_save(toTrainData), 0).clip(min=1)

    def rescale(raw_data):
        return (reshape_for_save(raw_data) - offset) / scale

    return rescale(toTrainData), rescale(toTestData)


def readCIFAR100():
    f = open('dataset/cifar100/train', 'rb')
    train_data_dict = pickle.load(f, encoding='iso-8859-1')
    f.close()

    X = train_data_dict['data']
    y = train_data_dict['fine_labels']

    f = open('dataset/cifar100/test', 'rb')
    test_data_dict = pickle.load(f, encoding='iso-8859-1')
    f.close()

    XTest = np.array(test_data_dict['data'])
    yTest = np.array(test_data_dict['fine_labels'])

    return X, y, XTest, yTest


def preprocessingMINST(toTrainData, toTestData):
    def reshape_for_save(raw_data):
        # raw_data = np.dstack(
        #     (raw_data[:, :784], raw_data[:, 784:1568], raw_data[:, 2048:]))
        raw_data = raw_data.reshape(
            (raw_data.shape[0], 28, 28, 1)).transpose(0, 3, 1, 2)
        return raw_data.astype(np.float32)

    offset = np.mean(reshape_for_save(toTrainData), 0)
    scale = np.std(reshape_for_save(toTrainData), 0).clip(min=1)

    def rescale(raw_data):
        return (reshape_for_save(raw_data) - offset) / scale

    return rescale(toTrainData), rescale(toTestData)


def preprocessingFashionMnist(toTrainData, toTestData):
    offset = np.mean(toTrainData, 0)
    scale = np.std(toTrainData, 0).clip(min=1)

    def rescale(raw_data):
        return (raw_data - offset) / scale

    return rescale(toTrainData), rescale(toTestData)


def readFLW():
    data_path = 'dataset'
    lfw_people = datasets.fetch_lfw_people(data_home=data_path, min_faces_per_person=40, resize=0.4)

    n_samples, h, w = lfw_people.images.shape  # resize=0.4 (1867,50,37)  # resize=1.0 (1867, 125, 94)
    print(lfw_people.images.shape)

    x = lfw_people.images.reshape(n_samples, 1, h, w) / 255.0
    y = lfw_people.target

    trainX, testX, trainY, testY = model_selection.train_test_split(x, y, test_size=.1, random_state=42)

    return trainX, trainY, testX, testY  # trainX(1680,1,50,37),testX(187,1,50,37)


def preprocessingLFW(toTrainData, toTestData):
    def reshape_for_save(raw_data):
        raw_data = np.dstack(
            (raw_data[:, :62500], raw_data[:, 62500:125000], raw_data[:, 125000:]))
        raw_data = raw_data.reshape(
            (raw_data.shape[0], 250, 250, 3)).transpose(0, 3, 1, 2)
        return raw_data.astype(np.float32)

    offset = np.mean(reshape_for_save(toTrainData), 0)
    scale = np.std(reshape_for_save(toTrainData), 0).clip(min=1)

    def rescale(raw_data):
        return (reshape_for_save(raw_data) - offset) / scale

    return rescale(toTrainData), rescale(toTestData)


def iterate_minibatches(inputs, targets, batch_size, shuffle=True):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)

    start_idx = None
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]

    if start_idx is not None and start_idx + batch_size < len(inputs):
        excerpt = indices[start_idx + batch_size:] if shuffle else slice(start_idx + batch_size, len(inputs))
        yield inputs[excerpt], targets[excerpt]


if __name__ == '__main__':

    # trainx, trainy, testx, testy = readFashionMnist()
    # trainx, trainy = shuffleData(trainx, trainy)
    # testx, testy = shuffleData(testx, testy)
    # trainx, testx = preprocessingFashionMnist(trainx, testx)
    trainset, testset = readImagenette()

    train_loader = DataLoader(
        trainset,
        batch_size=64,  # 500
        shuffle=True,
        num_workers=4,
        pin_memory=False
    )
    test_loader = DataLoader(
        testset,
        batch_size=64,  # 500
        shuffle=False,
        num_workers=4,
        pin_memory=False
    )

    epochs = 30
    device =   torch.device("cuda:3")

    # max_grad_norm_pool = [i for i in range(8, 50, 2)]
    # for max_grad_norm_ in max_grad_norm_pool:

    # net = cifar10_model()
    # net = CNN_Model()
    # net = BadNet()
    net = AlexNet()
    # net = ResNet(10, depth=18)
    # net = LeNet5()
    # net = ResNet(10, depth=18)
    # net = GoogLeNet(10)
    net.to(device)
    # for n, p in net.named_parameters():
    #     print(n, p.shape)
    # max_grad_norm_ = 2.0
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.93)

    # print(type(optimizer))
    batch_size = 64

    privacy_engine = PrivacyEngine(
        net,
        batch_size=batch_size,  # batch_size=256    epoch=90  lr=0.001
        sample_size=len(trainset.targets),
        alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
        noise_multiplier=0.6,  # 0.91 cifar 1.47 mnist  1.56 fashion
        max_grad_norm=6,  # 0.1 0.3 0.5 1.0
        secure_rng=False,
        target_delta=1e-5,
        # learning_rate=0.02,
        # device=device,
        # dynamic
        # C_0 = 4,
        # mu_0 = 1.099709491506219,
        # T = 4438,
        # rho_c = 2,
        # rho_mu = 2
    )
    privacy_engine.attach(optimizer)

    # print(optimizer.param_groups[0]["momentum"]);

    # DPFed
    # filename_max_grad_norm = './cifar10/ada/c_cifar10_lr1_momentum0.5_batchsize1024_sigma0.91_c10_garma_0.1_tanh_1.json'
    # filename_acc = './cifar10/ada/acc_cifar10_lr1_momentum0.5_batchsize1024_sigma0.91_c10_garma_0.1_tanh_1.json'
    # fed_norm = {}
    # fed_acc = {}
    # filename = 'positive_and_negative_c10.json'

    # yuda
    # gradNorm = {}
    # accuracy = {}
    # C_begin =5
    # filename = 'imagenette/resnet/yuda/clipping_values/imagenette_lr0.001_momentum0.9_batch64_sigma_0.6_relu_5_6.json'
    # filename_ = 'imagenette/resnet/yuda/acc/imagenette_lr0.001_momentum0.9_batch64_sigma_0.6_relu_5_6.json'

    # dynamic
    temp_loss = 0.0
    top1_acc = []
    epoch_acc = {}
    epoch_loss = {}
    gradNorm = {}
    accuracy = {}
    dynamic_c = {}
    dynamic_noise = {}

    # filename_dynamic_c = 'imagenette/resnet/papernot/clipping_values/resnet18_imagenette_lr0.001_momentum0.9_batchsize64_sigma0.6_c5_tanh_6'
    # filename_dynamic_noise = 'imagenette/resnet/papernot/noise/resnet18_imagenette_lr0.001_momentum0.9_batchsize64_sigma0.6_c5_tanh_6'
    # filename_epoch_acc = 'imagenette_mou/tanh/eps_46.6/alexnet_imagenette_lr0.001_momentum0.94_batchsize64_eps_46.6_c6_tanh_dynamic_0.93_dj1.json'
    filename_epoch_acc = '/kolla/lgb/dynamic_momentum/imagenette/tanh/dec_m_0.93_eps_11_C6.json'

    # Dict1 = {}
    # clip_value_ = 1.0
    # ACC = []

    # 初始动量设置 ： momentum = 0.99
    initialMomentum = 0.95
    # 动量缩放因子 scaleFactor = 0.9
    # mul = 0
    for epoch in range(epochs):

        net.train()
        # privacy_engine.max_grad_norm = C_begin / min(2, 1 + epoch / epochs)
        # gradNorm[epoch] = privacy_engine.max_grad_norm
        # print(optimizer.param_groups[0]["momentum"])
        # # if(epoch >= 5):
        initialMomentum -= 0.01
        print("------------------------------------------")
        optimizer.param_groups[0]["momentum"] = initialMomentum

        # elif(epoch >= 5 and epoch % 5 == 0):
        #     print("------------------------------------------")
        #     scaleFactor -+ = 0.02
        #     optimizer.param_groups[0]["momentum"] = initialMomentum * scaleFactor
        # 按epoch 改变 momentum
        # dynamic
        dynamic_c[epoch] = []
        dynamic_noise[epoch] = []

        # max_grad_norm = []

        # for x, y in iterate_minibatches(trainx, trainy, batch_size):
        for x, y in train_loader:
            x, y = torch.tensor(x).type(torch.float32), torch.tensor(y).type(torch.long)
            x, y = x.to(device), y.to(device)

            # dynamic_c[epoch].append(privacy_engine.max_grad_norm)
            # dynamic_noise[epoch].append(privacy_engine.noise_multiplier)

            optimizer.zero_grad()
            outputs = net(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            # max_grad_norm.append(privacy_engine.max_grad_norm)

            # # all_norms_.append(privacy_engine._PrivacyEngine_median.cpu().numpy())
            epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(target_delta=1e-5)
            print(
                # f"\tLearning rate: {ExpLR.get_lr()} \t"
                f"Train Epoch: {epoch} "
                # f"Loss: {np.mean(losses):.6f} "
                # f"Acc@1: {np.mean(top1_acc):.6f} "
                f"(epsilon = {epsilon:.4f}, delta = {1e-5}) for alpha = {best_alpha}"
            )
            #         # epsilon_[index] = epsilon
            # change_rate_.append(privacy_engine.change_rate_)
        # change_rate_ = np.array(change_rate_)
        # change_rate_ = np.median(change_rate_, axis=0)
        # # print(change_rate_)
        # layers = {}
        # for idx, rate in enumerate(change_rate_):
        #     layers[idx] = rate
        # Dict[epoch] = layers
        # fed_norm[epoch] = max_grad_norm

        net.eval()
        pred_y = []
        with torch.no_grad():
            # for x_, y in iterate_minibatches(testx, testy, batch_size, shuffle=False):
            for x_, _ in test_loader:
                x_ = torch.tensor(x_).type(torch.float32)
                x_ = x_.to(device)
                outputs = net(x_)
                pred_y.append(torch.max(outputs, 1)[1].data.cpu().numpy())

            pred_y = np.concatenate(pred_y)

        acc = accuracy_score(testset.targets, pred_y)
        # fed_acc[epoch] = acc
        accuracy[epoch] = acc
        # Dict1[epoch] = max_grad_norm
        print('Epoch {} Test Accuracy: {}'.format(epoch, acc))

        # with open(filename, 'w') as json_file:
        #      json.dump(gradNorm, json_file, indent=4)
        # with open(filename_, 'w') as json_file:
        #      json.dump(accuracy, json_file, indent=4)

        # with open(filename_max_grad_norm, 'w') as json_file:
        #      json.dump(fed_norm, json_file, indent=4)
        # with open(filename_acc, 'w') as json_file:
        #      json.dump(fed_acc, json_file, indent=4)
        # print(dynamic_c, dynamic_noise, ac)

        # with open(filename_dynamic_c, 'w') as json_file:
        #     json.dump(dynamic_c, json_file, indent=4)
        # with open(filename_dynamic_noise, 'w') as json_file:
        #     json.dump(dynamic_noise, json_file, indent=4)
        with open(filename_epoch_acc, 'w') as json_file:
            json.dump(accuracy, json_file, indent=4)

    # print(len(all_norms_))
    # all_norms_ = sorted(all_norms_)
    # size = len(all_norms_)
    # if size % 2 == 0:
    #     median_ = all_norms_[size // 2] / 2 + all_norms_[size // 2 - 1] / 2
    # else:
    #     median_ = all_norms_[(size - 1) // 2]
    # Dict[epoch] = acc
    # print('Epoch {} Median {}'.format(epoch, float(median_)))
    # with open(filename, 'w') as f:
    #     json.dump(Dict, f, indent=4)
    # with open(filename1, 'w') as f:
    #     json.dump(Dict1, f, indent=4)
    # for clip_value in clip_value_pool:
#
#         privacy_engine.max_grad_norm = clip_value
#         step = 0
#
#         for p in net.parameters():
#             if p.requires_grad:
#                 p.grad = params_grad[step]
#                 step += 1
#         net.load_state_dict(net_params)
#
#         net.train()
#         for index, (x, y) in enumerate(iterate_minibatches(trainx, trainy, batch_size)):
#             x, y = torch.tensor(x).type(torch.float32), torch.tensor(y).type(torch.long)
#             x, y = x.to(device), y.to(device)
#
#             optimizer.zero_grad()
#             outputs = net(x)
#             loss = criterion(outputs, y)
#             loss.backward()
#             optimizer.step()
#
#             # epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(target_delta=1e-5)
#             # print(
#             #     # f"\tLearning rate: {ExpLR.get_lr()} \t"
#             #     f"Train Epoch: {epoch} "
#             #     # f"Loss: {np.mean(losses):.6f} "
#             #     # f"Acc@1: {np.mean(top1_acc):.6f} "
#             #     f"(epsilon = {epsilon:.4f}, delta = {1e-5}), clip value = {clip_value} for alpha = {best_alpha}"
#             # )
#
#         net.eval()
#         pred_y = []
#         with torch.no_grad():
#             for x_, _ in iterate_minibatches(testx, testy, batch_size, shuffle=False):
#                 x_ = torch.tensor(x_).type(torch.float32)
#                 x_ = x_.to(device)
#                 outputs = net(x_)
#                 pred_y.append(torch.max(outputs, 1)[1].data.cpu().numpy())
#
#             pred_y = np.concatenate(pred_y)
#
#         acc = accuracy_score(testy, pred_y)
#         print('clip_value: {} acc : {}'.format(clip_value, acc))
#         # Dict[epoch] = acc
#
#         if acc < top_acc and abs(acc - top_acc) > 0.005:
#
#             privacy_engine.max_grad_norm = clip_value_chosen
#             step = 0
#             for p in net.parameters():
#                 if p.requires_grad:
#                     p.grad = params_grad[step]
#                     step += 1
#             net.load_state_dict(net_params)
#
#             net.train()
#             for index, (x, y) in enumerate(iterate_minibatches(trainx, trainy, batch_size)):
#                 x, y = torch.tensor(x).type(torch.float32), torch.tensor(y).type(torch.long)
#                 x, y = x.to(device), y.to(device)
#
#                 optimizer.zero_grad()
#                 outputs = net(x)
#                 loss = criterion(outputs, y)
#                 loss.backward()
#                 optimizer.step()
#             Dict[epoch] = (clip_value_chosen, top_acc)
#             print('Epoch : {} Batch {} Top Acc : {} c : {}'.format(epoch, index, top_acc, clip_value_chosen))
#             break
#         elif acc > top_acc:
#             top_acc = acc
#             clip_value_chosen = clip_value
#             print('Epoch {} Acc {}'.format(epoch, acc))
#
# filename = 'fashionmnist_lr{}_batchsize{}_sigma{}.json'.format(1, 1024, 1.47)
# with open(filename, 'w') as f:
#     json.dump(Dict, f, indent=4)

# for clip_value in clip_value_pool:
#
#     privacy_engine.max_grad_norm = clip_value
#     step = 0
#
#     for p in net.parameters():
#         if p.requires_grad:
#             p.grad = params_grad[step]
#             step += 1
#
#     optimizer.zero_grad()
#     outputs = net(x)
#     loss = criterion(outputs, y)
#     loss.backward()
#     optimizer.step()
#
#     net.eval()
#     pred_y = []
#     with torch.no_grad():
#         for x_, _ in iterate_minibatches(testx, testy, batch_size, shuffle=False):
#             x_ = torch.tensor(x_).contiguous()
#             x_ = x_.to(device)
#             outputs = net(x_)
#             pred_y.append(torch.max(outputs, 1)[1].data.cpu().numpy())
#
#         pred_y = np.concatenate(pred_y)
#
#     acc = accuracy_score(testy, pred_y)
#     # print('Epoch {} Test Accuracy: {}  c : {}'.format(epoch, accuracy_score(testy, pred_y), clip_value))
#
#     epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(target_delta=1e-5)
#     print(
#         # f"\tLearning rate: {ExpLR.get_lr()} \t"
#         f"Train Epoch: {epoch} "
#         # f"Loss: {np.mean(losses):.6f} "
#         # f"Acc@1: {np.mean(top1_acc):.6f} "
#         f"(epsilon = {epsilon:.4f}, delta = {1e-5}), clip value = {clip_value} for alpha = {best_alpha}"
#     )
#     net.train()
#     # break
#     if acc < top_acc and abs(acc - top_acc) > 0.005:
#         # chosen_clipValue.append(clip_value)
#         # top_acc.append(temp_acc)
#         privacy_engine.max_grad_norm = clip_value_chosen
#         step = 0
#         for p in net.parameters():
#             if p.requires_grad:
#                 p.grad = params_grad[step]
#                 step += 1
#
#         optimizer.zero_grad()
#         outputs = net(x)
#         loss = criterion(outputs, y)
#         loss.backward()
#         optimizer.step()
#
#         print('Epoch : {} Batch {} Top Acc : {} c : {}'.format(epoch, index, top_acc, clip_value_chosen))
#         break
#     elif acc > top_acc:
#         top_acc = acc
#         clip_value_chosen = clip_value

#     epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(target_delta=1e-5)
#     print(
#         # f"\tLearning rate: {ExpLR.get_lr()} \t"
#         f"Train Epoch: {epoch} "
#         # f"Loss: {np.mean(losses):.6f} "
#         # f"Acc@1: {np.mean(top1_acc):.6f} "
#         f"(epsilon = {epsilon:.4f}, delta = {1e-5}) for alpha = {best_alpha}"
#     )
#
# net.eval()
# pred_y = []
# with torch.no_grad():
#     for x, _ in iterate_minibatches(testx, testy, batch_size, shuffle=False):
#         x = torch.tensor(x).contiguous()
#         x = x.to(device)
#
#         outputs = net(x)
# Ss z
#         pred_y.append(torch.max(outputs, 1)[1].data.cpu().numpy())
#
#     pred_y = np.concatenate(pred_y)
#
# print('Epoch {} Test Accuracy: {}'.format(epoch, accuracy_score(testy, pred_y)))
