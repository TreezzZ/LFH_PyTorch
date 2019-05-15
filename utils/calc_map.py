#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np


def calc_map(query_code, database_code, query_labels, database_labels):
    """计算mAP

    Parameters
        query_code: ndarray, {-1, +1}^{m * Q}
        query的hash code

        database_code: ndarray, {-1, +1}^{n * Q}
        database的hash code

        query_labels: ndarray, {0, 1}^{m * n_classes}
        query的label，onehot编码

        database_labels: ndarray, {0, 1}^{n * n_classes}
        database的label，onehot编码

    Returns
        meanAP: float
        Mean Average Precision
    """
    num_query = query_labels.shape[0]
    mean_AP = 0.0
    pre_calc_retrieval = (query_labels @ database_labels.T > 0).astype(np.float32)
    pre_calc_retrieval_cnt = pre_calc_retrieval.sum(axis=1)
    pre_calc_hamming_dist = 0.5 * (query_code.shape[1] - query_code @ database_code.T)
    for i in range(num_query):
        # 检索
        retrieval = pre_calc_retrieval[i, :]

        # 检索到数量
        retrieval_cnt = pre_calc_retrieval_cnt[i]

        # 未检索到
        if retrieval_cnt == 0:
            continue

        # hamming distance
        hamming_dist = pre_calc_hamming_dist[i, :]

        # 根据hamming distance安排检索结果位置
        retrieval = retrieval[np.argsort(hamming_dist)]

        # 每个位置打分
        score = np.linspace(1, retrieval_cnt, retrieval_cnt)

        # 检索到的下标位置
        index = np.asarray(np.where(retrieval == 1)) + 1.0

        mean_AP += np.mean(score / index)

    mean_AP = mean_AP / num_query
    return mean_AP


