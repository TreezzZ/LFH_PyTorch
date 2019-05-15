# -*- coding: utf-8 -*-


from data.onehot import encode_onehot

import numpy as np
import torch
import scipy.io as sio

import os
import sys
import pickle


def load_data(path, train=True):
    """加载CIFAR 10数据集

    Parameters
        path: str
        数据集路径

        train: bool
        True: 加载训练数据; False: 加载测试数据

    Returns
        data: ndarray
        数据

        labels: ndarray
        标签
    """
    base_folder = 'cifar-10-batches-py'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    if train:
        train_data = []
        train_labels = []
        for fentry in train_list:
            f = fentry[0]
            file = os.path.join(os.path.expanduser(path), base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            train_data.append(entry['data'])
            if 'labels' in entry:
                train_labels += entry['labels']
            else:
                train_labels += entry['fine_labels']
            fo.close()

        train_data = np.concatenate(train_data)
        train_data = train_data.reshape((50000, 3, 32, 32))
        train_data = train_data.transpose((0, 2, 3, 1))  # convert to HWC

        return train_data, train_labels
    else:
        f = test_list[0][0]
        file = os.path.join(os.path.expanduser(path), base_folder, f)
        fo = open(file, 'rb')
        if sys.version_info[0] == 2:
            entry = pickle.load(fo)
        else:
            entry = pickle.load(fo, encoding='latin1')
        test_data = entry['data']
        if 'labels' in entry:
            test_labels = entry['labels']
        else:
            test_labels = entry['fine_labels']
        fo.close()
        test_data = test_data.reshape((10000, 3, 32, 32))
        test_data = test_data.transpose((0, 2, 3, 1))  # convert to HWC

        return test_data, test_labels


def load_data_gist(path, train=True):
    """加载对cifar10使用gist提取的数据

    Parameters
        path: str
        数据路径

        train: bools
        True，加载训练数据; False，加载测试数据

    Returns
        data: ndarray
        数据

        labels: ndarray
        标签
    """
    mat_data = sio.loadmat(path)

    if train:
        data = mat_data['traindata']
        labels = mat_data['traingnd'].astype(np.int)
    else:
        data = mat_data['testdata']
        labels = mat_data['testgnd'].astype(np.int)

    return data, labels
