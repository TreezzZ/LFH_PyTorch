#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np


def sampling(num_dataset, num_samples):
    """从num_dataset条数据中随机采样num_samples条

    Parameters
        num_dataset: int
        原始数据大小

        num_samples: int
        采样数据大小

    Returns
        sample_index: Tensor
        采样数据下标
    """
    #sample_index = torch.randperm(num_dataset)[:num_samples]
    return list(np.random.permutation(num_dataset))[:num_samples]
