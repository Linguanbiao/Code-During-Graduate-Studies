import torch
from torchvision import datasets as dset
from torchvision import transforms
from torch.utils.data import DataLoader, RandomSampler
from torch import nn
import torch.optim as optim
from sklearn import datasets

from torchvision import datasets

import numpy as np
import copy
import pickle
import random
from dataset import readcifar10_GAN, readCIFAR10, readCIFAR100, readMINST, readNews, readFLW, readLocation_train, \
    readLocation_test, \
    readPurchase10_test, readPurchase10_train, readPurchase50_test, readPurchase50_train, readPurchase2_train, \
    readPurchase2_test, readPurchase20_train, readPurchase20_test, readPurchase100_test, readPurchase100_train
    
from preprocessing import preprocessingCIFAR, preprocessingCIFAR_GAN, preprocessingMINST, preprocessingNews, \
    preprocessingLFW

from sklearn.metrics import accuracy_score
from opacus import PrivacyEngine
from net.CNN import CNN_Model
from net.CIFAR import cifar10_model
from net.simple_CNN import cnn1
import warnings

warnings.filterwarnings("ignore")


def readCIFAR10():
    data_path = 'data/cifar-10-batches-py'
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

def readFashionMnist():
    data_path = 'dataset'
    train = datasets.FashionMNIST(data_path, train=True, download=True)
    test = datasets.FashionMNIST(data_path, train=False, download=False)
    x, y = train.data.numpy(), train.targets.numpy()
    xTest, yTest = test.data.numpy(), test.targets.numpy()
    x = x[:, np.newaxis, :, :]
    xTest = xTest[:, np.newaxis, :, :]
    return x, y, xTest, yTest

def readCIFAR100():
    f = open('dataset/cifar-100-python/train', 'rb')
    train_data_dict = pickle.load(f, encoding='iso-8859-1')
    f.close()

    X = train_data_dict['data']
    y = train_data_dict['fine_labels']

    f = open('dataset/cifar-100-python/test', 'rb')
    test_data_dict = pickle.load(f, encoding='iso-8859-1')
    f.close()

    XTest = np.array(test_data_dict['data'])
    yTest = np.array(test_data_dict['fine_labels'])

    return X, y, XTest, yTest
    

def shuffleData(datax, datay):
    c = list(zip(datax, datay))
    random.shuffle(c)
    dataX, dataY = zip(*c)
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

def preprocessingFashionMnist(toTrainData, toTestData):
    offset = np.mean(toTrainData, 0)
    scale = np.std(toTrainData, 0).clip(min=1)

    def rescale(raw_data):
        return (raw_data - offset) / scale

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

    trainx, trainy, testx, testy = reedCIFAR10()
    trainx, trainy = shuffleData(trainx, trainy)
    testx, testy = shuffleData(testx, testy)
    trainx, testx = preprocessingFashionMnist(trainx, testx)

    epochs = 40
    device = torch.device("cuda:0")
    n_in = trainx.shape
    n_out = len(np.unique(trainy))
    # net = cifar10_model()
    net = cnn1()
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=1.0, momentum=0.5)
    batch_size = 1024

    privacy_engine = PrivacyEngine(
        net,
        batch_size=batch_size,  # batch_size=256    epoch=90  lr=0.001
        sample_size=len(trainx),
        alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
        noise_multiplier=1.55,  # 1.1
        max_grad_norm=1.0,  # 1.0
        secure_rng=False,
        target_delta=1e-5
    )
    privacy_engine.attach(optimizer)
    clip_value_pool = [0.05 + x / 100.0 for x in range(0, 1000, 2)]
    Dict = {}
    # clip_value_ = 1.0

    for epoch in range(epochs):
        params_grad = [p.grad for p in net.parameters() if p.requires_grad]
        net_params = copy.deepcopy(net.state_dict())
        top_acc = 0
        clip_value_chosen = -1
        # if epoch != 0 and epoch % 5 == 0:
        #     clip_value_ *= 0.8
        #     privacy_engine.max_grad_norm = clip_value_
        #
        # net.train()
        # for index, (x, y) in enumerate(iterate_minibatches(trainx, trainy, batch_size)):
        #     x, y = torch.tensor(x).contiguous(), torch.tensor(y).type(torch.long)
        #     x, y = x.to(device), y.to(device)
        #
        #     optimizer.zero_grad()
        #     outputs = net(x)
        #     loss = criterion(outputs, y)
        #     loss.backward()
        #     optimizer.step()
        #
        #     epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(target_delta=1e-5)
        #     print(
        #         # f"\tLearning rate: {ExpLR.get_lr()} \t"
        #         f"Train Epoch: {epoch} "
        #         # f"Loss: {np.mean(losses):.6f} "
        #         # f"Acc@1: {np.mean(top1_acc):.6f} "
        #         f"(epsilon = {epsilon:.4f}, delta = {1e-5}), clip value = {clip_value_} for alpha = {best_alpha}"
        #     )
        #
        # net.eval()
        # pred_y = []
        # with torch.no_grad():
        #     for x_, _ in iterate_minibatches(testx, testy, batch_size, shuffle=False):
        #         x_ = torch.tensor(x_).contiguous()
        #         x_ = x_.to(device)
        #         outputs = net(x_)
        #         pred_y.append(torch.max(outputs, 1)[1].data.cpu().numpy())
        #
        #     pred_y = np.concatenate(pred_y)
        #
        # acc = accuracy_score(testy, pred_y)
        # print('Acc : ', acc)

        for clip_value in clip_value_pool:

            privacy_engine.max_grad_norm = clip_value
            step = 0

            for p in net.parameters():
                if p.requires_grad:
                    p.grad = params_grad[step]
                    step += 1
            # print('net_params', net_params)
            net.load_state_dict(net_params)

            net.train()
            for index, (x, y) in enumerate(iterate_minibatches(trainx, trainy, batch_size)):

                x, y = torch.tensor(x).contiguous().type(torch.float32), torch.tensor(y).type(torch.long)
                x, y = x.to(device), y.to(device)

                optimizer.zero_grad()
                outputs = net(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

                epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(target_delta=1e-5)
                print(
                    # f"\tLearning rate: {ExpLR.get_lr()} \t"
                    f"Train Epoch: {epoch} "
                    # f"Loss: {np.mean(losses):.6f} "
                    # f"Acc@1: {np.mean(top1_acc):.6f} "
                    f"(epsilon = {epsilon:.4f}, delta = {1e-5}), clip value = {clip_value} for alpha = {best_alpha}"
                )

            net.eval()
            pred_y = []
            with torch.no_grad():
                for x_, _ in iterate_minibatches(testx, testy, batch_size, shuffle=False):
                    x_ = torch.tensor(x_).contiguous().type(torch.float32)
                    x_ = x_.to(device)
                    outputs = net(x_)
                    pred_y.append(torch.max(outputs, 1)[1].data.cpu().numpy())

                pred_y = np.concatenate(pred_y)

            acc = accuracy_score(testy, pred_y)
            print('Acc : ', acc)

            if acc < top_acc and abs(acc - top_acc) > 0.005:

                privacy_engine.max_grad_norm = clip_value_chosen
                step = 0
                for p in net.parameters():
                    if p.requires_grad:
                        p.grad = params_grad[step]
                        step += 1

                net.train()
                for index, (x, y) in enumerate(iterate_minibatches(trainx, trainy, batch_size)):

                    x, y = torch.tensor(x).contiguous().type(torch.float32), torch.tensor(y).type(torch.long)
                    x, y = x.to(device), y.to(device)

                    optimizer.zero_grad()
                    outputs = net(x)
                    loss = criterion(outputs, y)
                    loss.backward()
                    optimizer.step()
                Dict[epoch] = clip_value_chosen
                print('Epoch : {} Batch {} Top Acc : {} c : {}'.format(epoch, index, top_acc, clip_value_chosen))
                break

            elif acc > top_acc:
                top_acc = acc
                clip_value_chosen = clip_value
                print('Epoch {} Acc {}'.format(epoch, acc))

    import json
    jsonstr = json.dumps(Dict)
    filename = open('log_mnist.txt', 'w')  # dict转josn
    filename.write(jsonstr)

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