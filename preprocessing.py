# ！-*- coding:utf-8 -*-
import numpy as np


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

# def preprocessingCIFAR_GAN(Data,toTestData):
#
#     offset = np.mean(Data, 0)
#     scale = np.std(Data, 0).clip(min=1)
#
#     def rescale(raw_data):
#         return (raw_data - offset) / scale
#
#     return rescale(Data),rescale(toTestData)


def preprocessingCIFAR_GAN(toTrainData, toTestData):
    # def reshape_for_save(raw_data):
    #     raw_data = np.dstack(
    #         (raw_data[:, :1024], raw_data[:, 1024:2048], raw_data[:, 2048:]))
    #     raw_data = raw_data.reshape(
    #         (raw_data.shape[0], 32, 32, 3)).transpose(0, 3, 1, 2)
    #     return raw_data.astype(np.float32)

    offset = np.mean(toTrainData, 0)
    scale = np.std(toTrainData, 0).clip(min=1)

    def rescale(raw_data):
        return (raw_data - offset) / scale

    return rescale(toTrainData), rescale(toTestData)


def preprocessingNews(toTrainData, toTestData):
    def normalizeData(X):
        offset = np.mean(X, 0)
        scale = np.std(X, 0).clip(min=1)
        X = (X - offset) / scale
        X = X.astype(np.float32)
        return X

    return normalizeData(toTrainData), normalizeData(toTestData)


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


def preprocessingAdult(toTrainData, toTestData):
    def normalizeData(X):
        offset = np.mean(X, 0)
        scale = np.std(X, 0).clip(min=1)
        X = (X - offset) / scale
        X = X.astype(np.float32)
        return X

    return normalizeData(toTrainData), normalizeData(toTestData)


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
