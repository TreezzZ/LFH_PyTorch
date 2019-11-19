import numpy as np
import scipy.io as sio

from data.transform import encode_onehot


def load_data_gist(root):
    """
    Load cifar10-gist dataset.

    Args
        root(str): Path of dataset.

    Returns
        query_data(numpy.ndarray, num_query*512): Query data.
        query_targets(numpy.ndarray, num_query*10): One-hot query targets.
        train_data(numpy.ndarray, num_train*512): Training data.
        train_targets(numpy.ndarray, num_train*10): One-hot training targets.
    """
    # Load dta
    mat_data = sio.loadmat(root)
    query_data = mat_data['testdata']
    query_targets = mat_data['testgnd'].astype(np.int)
    train_data = mat_data['traindata']
    train_targets = mat_data['traingnd'].astype(np.int)

    # Data pre-process
    # One-hot
    query_targets = encode_onehot(query_targets)
    train_targets = encode_onehot(train_targets)

    # Normalization
    data = np.concatenate((query_data, train_data), axis=0)
    data = (data - data.mean()) / data.std()
    query_data = data[:query_data.shape[0], :]
    train_data = data[query_data.shape[0]:, :]

    return query_data, query_targets, train_data, train_targets
