#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from tqdm import tqdm
from utils.sampling import sampling
import utils.out_of_sample_extension as ose
from utils.calc_map import calc_map
from utils.cal_similarity_matrix import cal_similarity_matrix


def lfh(code_length,
        num_samples,
        T,
        beta,
        epsilon,
        train_data,
        train_labels,
        test_data,
        test_labels,
        lamda,
        ):
    """LFH algorithm
    
    Parameters
        code_length: int
        Binary hash code length

        num_samples: int
        采样数量

        T: int
        Max iteration

        beta: float
        Hyper parameter

        epsilon: float
        Hyper parameter

        train_data: Tensor
        训练数据

        lamda: float
        Hyper parameter

    Returns
        U: ndarray
        学到的hash code

        meanAP: float
        Mean Average Precision
    """
    # 计算相似度矩阵
    S = cal_similarity_matrix(train_labels, train_labels).float()

    N = S.shape[0]
    pre_calc = torch.inverse(train_data.t() @ train_data +
                             lamda * torch.eye(train_data.shape[1])) @ train_data.t()
    # 初始化U
    U = torch.randn(N, code_length)

    for iteration in tqdm(range(T)):
        U = _update_U(U, S, num_samples, code_length, beta)

    U = U.sign()

    meanAP = evaluate(test_data, test_labels, pre_calc, train_labels, U)
    return U, meanAP


def _update_U(U, S, num_samples, code_length, beta):
    """按行更新U

    Parameters
        U: Tensor
        见论文
        
        S: Tensor
        Similarity matrix

        num_samples: int
        Number of samples

        code_length: int
        binary hash code length

        beta: float
        Hyper Parameter

    Returns
        U: ndarray
        database hash code
    """
    # 数据集大小
    num_dataset = U.shape[0]
    
    # sampling
    sample_index = sampling(num_dataset, num_samples)

    theta = (U @ U[sample_index, :].t()) / 2
    sigmoid = 1. / (1 + torch.exp(-theta))

    # 计算Hessian矩阵
    H = -U[sample_index, :].t() @ U[sample_index, :] / 8 - torch.eye(code_length) / beta

    # 计算一阶导
    du = (S[:, sample_index] - sigmoid) @ U[sample_index, :] - U / beta

    # 更新U
    U = U - du @ torch.inverse(H)

    return U


def evaluate(test_data, test_labels, pre_calc, train_labels, U):
    """评估算法

    Parameters
        test_data: Tensor
        测试数据

        test_labels: Tensor
        测试标签

        train_data: Tensor
        训练数据

        train_labels: Tensor
        训练标签

        U: ndarray
        LFH学到的hash code

        lamda: float
        正则化参数

    Returns
        meanAP: float
        mean Average precision
    """
    # 训练一个线性分类器
    W = ose.linear_extension(pre_calc, U)

    # 将query points映射为hash code
    outputs = (test_data @ W).sign()

    # 计算map
    meanAP = calc_map(outputs.numpy(), U.numpy(), test_labels.numpy(), train_labels.numpy())

    return meanAP
