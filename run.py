#!/usr/bin/env python
# -*- coding: utf-8 -*-

import LFH

from torch.utils.tensorboard import SummaryWriter
import data.dataloader as DataLoader
import argparse


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
    opt.lamda = opt.lamda * train_data.shape[0] / train_data.shape[1]
    opt.beta = opt.beta / opt.code_length

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
    parser.add_argument('--num-samples', default='200', type=int,
                        help='hyper-parameter: numbers of sampling data (default: 200)')
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

    # 可视化
    writer = SummaryWriter()
    meanAP = run_lfh(opt)
    writer.add_scalar('mAP', meanAP)
