#!/usr/bin/env python
# -*- coding: utf-8 -*-

import LFH
import data.dataloader as DataLoader

from torch.utils.tensorboard import SummaryWriter
from loguru import logger

import argparse
import os
import torch
import datetime


def run_lfh(opt):
    """运行LFH算法

    Parameters
        opt: parser
        程序运行参数

    Returns
        meanAP: float
        Mean Average Precision
    """
    # 加载数据
    train_data, train_labels = DataLoader.load_data(path=opt.data_path,
                                                    dataset='cifar10_gist',
                                                    train=True,
                                                    )

    test_data, test_labels = DataLoader.load_data(path=opt.data_path,
                                                  dataset='cifar10_gist',
                                                  train=False,
                                                  )

    # 正则化超参
    lamda = opt.lamda * train_data.shape[0] / train_data.shape[1]
    beta = opt.beta / opt.code_length

    # LFH算法
    U, meanAP = LFH.lfh(code_length=opt.code_length,
                        num_samples=opt.code_length,
                        T=opt.max_iterations,
                        beta=opt.beta,
                        epsilon=opt.epsilon,
                        train_data=train_data,
                        train_labels=train_labels,
                        test_data=test_data,
                        test_labels=test_labels,
                        lamda=opt.lamda,
                        )

    logger.info("hyper-parameters: code_length: {}, num_samples: {}, max_iterations: {}"
                ", beta: {}, epsilon: {}, lamda: {}".format(opt.code_length,
                                                            opt.num_samples,
                                                            opt.max_iterations,
                                                            opt.beta,
                                                            opt.epsilon,
                                                            opt.lamda),
                )
    logger.info("meanAP: {:.4f}".format(meanAP))

    # 保存结果
    torch.save(U, os.path.join('result',
                               '{}_{}_{}_{}_{:.4f}.t'.format(datetime.datetime
                                                             .now().strftime('%Y_%m_%d_%H_%M_%S'),
                                                             opt.code_length,
                                                             opt.code_length,
                                                             opt.max_iterations,
                                                             meanAP,
                                                             )))

    return meanAP


def load_parse():
    """加载程序参数

    Parameters
        None

    Returns
        opt: parser
        程序参数
    """
    parser = argparse.ArgumentParser(description='LFH')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        help='dataset used to train (default: cifar10)')
    parser.add_argument('--data-path', default='~/data/pytorch_cifar10', type=str,
                        help='path of cifar10 dataset (default: ~/data/pytorch_cifar10')

    parser.add_argument('--code-length', default='64', type=int,
                        help='hyper-parameter: binary hash code length (default: 64)')
    parser.add_argument('--num-samples', default='64', type=int,
                        help='hyper-parameter: numbers of sampling data (default: same as code-length)')
    parser.add_argument('--max-iterations', default='50', type=int,
                        help='hyper-parameter: numbers of iterations (default: 50)')
    parser.add_argument('--beta', default='30', type=float,
                        help='hyper-parameter: beta (default: 30)')
    parser.add_argument('--epsilon', default='0.1', type=float,
                        help='hyper-parameter: value of when to stop compute (default: 0.1)')
    parser.add_argument('--lamda', default=1, type=float,
                        help='hyper-parameter: regularization term (default: 1)')

    return parser.parse_args()


if __name__ == "__main__":
    opt = load_parse()
    writer = SummaryWriter()
    logger.add('logs/file_{time}.log')

    code_lengths = [8, 16, 24, 32, 48, 64, 96, 128]

    for i in code_lengths:
        # 可视化
        opt.code_length = i
        meanAP = run_lfh(opt)
        writer.add_scalar('mAP', meanAP, i)

    writer.close()
