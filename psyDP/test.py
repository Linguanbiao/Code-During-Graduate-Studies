import torch
from torchvision import datasets as dset
from torchvision import transforms
from torch.utils.data import DataLoader, RandomSampler
from torch import nn
import torch.optim as optim

import numpy as np
import copy
import pickle
import random

from sklearn.metrics import accuracy_score
from opacus import PrivacyEngine
from model.cnn import CNN_Model
from sklearn import model_selection, datasets
from torchvision import datasets
from model.cifar10_model import cifar10_model
from model.cnn1 import cnn1
import warnings

warnings.filterwarnings("ignore")


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

    trainx, trainy, testx, testy = readFashionMnist()
    trainx, trainy = shuffleData(trainx, trainy)
    testx, testy = shuffleData(testx, testy)
    trainx, testx = preprocessingFashionMnist(trainx, testx)

    epochs = 40
    device = torch.device("cuda:1")
    net = cnn1()
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=1, momentum=0.5)
    batch_size = 1024
    all_norms_ = []

    privacy_engine = PrivacyEngine(
        net,
        batch_size=batch_size,  # batch_size=256    epoch=90  lr=0.001
        sample_size=len(trainx),
        alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
        noise_multiplier=1.47,  # 0.91 cifar 1.47 mnist  1.56 fashion
        max_grad_norm=1,  # 0.1 0.3 0.5 1.0
        secure_rng=False,
        target_delta=1e-5,
        all_norms_=all_norms_
    )
    privacy_engine.attach(optimizer)

    import json

    #
    #
    # def fun(x):
    #     y = -6.89087474e-05 * x ** 3 + 5.63497191e-03 * x ** 2 + -1.53736078e-01 * x ** 1 \
    #         + 2.13689750e+00
    #     return y
    #
    # with open('log_mnist_3.0.txt', 'r', encoding='utf8') as fp:
    #     json_data_1 = json.load(fp)
    # clip_value_pool = []
    # for i in json_data_1.values():
    #     clip_value_pool.append(i[0])
    clip_value_pool = [0.05 + x / 100.0 for x in range(0, 1000, 2)]
    Dict = {}
    # clip_value_ = 1.0
    # ACC = []
    for epoch in range(epochs):
        params_grad = [p.grad for p in net.parameters() if p.requires_grad]
        net_params = copy.deepcopy(net.state_dict())
        top_acc = 0
        clip_value_chosen = -1

    #     privacy_engine.max_grad_norm = fun(epoch)
    #     clip_value_ = fun(epoch)
    #     net.train()
    #     epsilon_ = {}
    #     for index, (x, y) in enumerate(iterate_minibatches(trainx, trainy, batch_size)):
    #         x, y = torch.tensor(x).type(torch.float32), torch.tensor(y).type(torch.long)
    #         x, y = x.to(device), y.to(device)
    #
    #         optimizer.zero_grad()
    #         outputs = net(x)
    #         loss = criterion(outputs, y)
    #         loss.backward()
    #         optimizer.step()
    #
    #         epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(target_delta=1e-5)
    #         print(
    #             # f"\tLearning rate: {ExpLR.get_lr()} \t"
    #             f"Train Epoch: {epoch} "
    #             # f"Loss: {np.mean(losses):.6f} "
    #             # f"Acc@1: {np.mean(top1_acc):.6f} "
    #             f"(epsilon = {epsilon:.4f}, delta = {1e-5}), clip value = {clip_value_} for alpha = {best_alpha}"
    #         )
    #         epsilon_[index] = epsilon
    #
    #     net.eval()
    #     pred_y = []
    #     with torch.no_grad():
    #         for x_, _ in iterate_minibatches(testx, testy, batch_size, shuffle=False):
    #             x_ = torch.tensor(x_).type(torch.float32)
    #             x_ = x_.to(device)
    #             outputs = net(x_)
    #             pred_y.append(torch.max(outputs, 1)[1].data.cpu().numpy())
    #
    #         pred_y = np.concatenate(pred_y)
    #
    #     acc = accuracy_score(testy, pred_y)
    #     print('Epoch {} Test Accuracy: {}'.format(epoch, acc))
    #     Dict[epoch] = {'eps': epsilon_, 'acc': acc}
    # with open('fashion2mnist_1.47.json', 'w') as f:
    #     json.dump(Dict, f, indent=4)
        for clip_value in clip_value_pool:

            privacy_engine.max_grad_norm = clip_value
            step = 0

            for p in net.parameters():
                if p.requires_grad:
                    p.grad = params_grad[step]
                    step += 1
            net.load_state_dict(net_params)

            net.train()
            for index, (x, y) in enumerate(iterate_minibatches(trainx, trainy, batch_size)):
                x, y = torch.tensor(x).type(torch.float32), torch.tensor(y).type(torch.long)
                x, y = x.to(device), y.to(device)

                optimizer.zero_grad()
                outputs = net(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

                epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(target_delta=1e-5)
                # print(
                #     # f"\tLearning rate: {ExpLR.get_lr()} \t"
                #     f"Train Epoch: {epoch} "
                #     # f"Loss: {np.mean(losses):.6f} "
                #     # f"Acc@1: {np.mean(top1_acc):.6f} "
                #     f"(epsilon = {epsilon:.4f}, delta = {1e-5}), clip value = {clip_value} for alpha = {best_alpha}"
                # )

            net.eval()
            pred_y = []
            with torch.no_grad():
                for x_, _ in iterate_minibatches(testx, testy, batch_size, shuffle=False):
                    x_ = torch.tensor(x_).type(torch.float32)
                    x_ = x_.to(device)
                    outputs = net(x_)
                    pred_y.append(torch.max(outputs, 1)[1].data.cpu().numpy())

                pred_y = np.concatenate(pred_y)

            acc = accuracy_score(testy, pred_y)
            print('clip_value: {} acc : {}'.format(clip_value, acc))
            # Dict[epoch] = acc

            if acc < top_acc and abs(acc - top_acc) > 0.005:

                privacy_engine.max_grad_norm = clip_value_chosen
                step = 0
                for p in net.parameters():
                    if p.requires_grad:
                        p.grad = params_grad[step]
                        step += 1
                net.load_state_dict(net_params)

                net.train()
                for index, (x, y) in enumerate(iterate_minibatches(trainx, trainy, batch_size)):
                    x, y = torch.tensor(x).type(torch.float32), torch.tensor(y).type(torch.long)
                    x, y = x.to(device), y.to(device)

                    optimizer.zero_grad()
                    outputs = net(x)
                    loss = criterion(outputs, y)
                    loss.backward()
                    optimizer.step()
                Dict[epoch] = (clip_value_chosen, top_acc)
                print('Epoch : {} Batch {} Top Acc : {} c : {}'.format(epoch, index, top_acc, clip_value_chosen))
                break
            elif acc > top_acc:
                top_acc = acc
                clip_value_chosen = clip_value
                print('Epoch {} Acc {}'.format(epoch, acc))

    filename = 'fashionmnist_lr{}_batchsize{}_sigma{}.json'.format(1, 1024, 1.47)
    with open(filename, 'w') as f:
        json.dump(Dict, f, indent=4)

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
    #
    #         pred_y.append(torch.max(outputs, 1)[1].data.cpu().numpy())
    #
    #     pred_y = np.concatenate(pred_y)
    #
    # print('Epoch {} Test Accuracy: {}'.format(epoch, accuracy_score(testy, pred_y)))