#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import torch


def cal_similarity_matrix(labels1, labels2):
    """计算similarity matrix

    Parameters
        labels1, labels2: Tensor
        标签

    Returns
        similarity_matrix: Tensor
        相似度矩阵
    """
    return labels1 @ labels2.t()
