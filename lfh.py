import torch

from utils.evaluate import mean_average_precision


def train(
        query_data,
        query_targets,
        train_data,
        train_targets,
        code_length,
        num_samples,
        max_iter,
        beta,
        lamda,
        topk,
):
    """
    Training model

    Args
        query_data(torch.Tensor, num_query*512): Query data.
        query_targets(torch.Tensor, num_query*10): One-hot query targets.
        train_data(torch.Tensor, num_train*512): Training data.
        train_targets(torch.Tensor, num_train*10): One-hot training targets.
        code_length(int): Hash code length.
        num_samples(int): Number of samples.
        max_iter(int): Number of iterations.
        beta, lamda(float): Hyper-parameters.
        topk(int): Calculate top k data map.

    Returns
        mAP(float): Mean Average Precision.
    """
    # Initialization
    N = train_data.shape[0]
    U = torch.randn(N, code_length)
    W_prime = torch.inverse(train_data.t() @ train_data +
                            lamda * torch.eye(train_data.shape[1])) @ train_data.t()

    # Compute similarity matrix
    S = (train_targets @ train_targets.t() > 0).float()

    # Training
    for i in range(max_iter):
        # Sample data from training dataset
        sample_index = torch.randperm(N)[:num_samples]

        # Compute theta and sigmoid
        theta_sigmoid = ((U @ U[sample_index, :].t()) / 2).sigmoid()

        # Compute Hessian matrix
        H = -U[sample_index, :].t() @ U[sample_index, :] / 8 - torch.eye(code_length) / beta

        # Compute first derivative
        du = (S[:, sample_index] - theta_sigmoid) @ U[sample_index, :] - U / beta

        # Update U
        U = U - du @ torch.inverse(H)

    # Generate retrieval dataset code
    retrieval_code = U.sign()

    # Out-of-sample extension
    W = W_prime @ retrieval_code

    # Generate query dataset code
    query_code = (query_data @ W).sign()

    # Evaluate
    mAP = mean_average_precision(
        query_code,
        retrieval_code,
        query_targets,
        train_targets,
        topk,
    )

    return mAP
