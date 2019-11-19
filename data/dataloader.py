import data.cifar10 as cifar10


def load_data(dataset, root):
    """
    Load dataset.

    Args
        dataset(str): Dataset name.
        root(str): Path of dataset.
    """
    if dataset == 'cifar10-gist':
        return cifar10.load_data_gist(root)
    else:
        raise ValueError('Invalid dataset name!')
