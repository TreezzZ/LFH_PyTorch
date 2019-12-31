import torch

from utils.evaluate import mean_average_precision


def train(
        train_data,
        train_targets,
        query_data,
        query_targets,
        retrieval_data,
        retrieval_targets,
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
        train_data(torch.Tensor): Training data.
        train_targets(torch.Tensor): One-hot training targets.
        query_data(torch.Tensor): Query data.
        query_targets(torch.Tensor): One-hot query targets.
        retrieval_data(torch.Tensor): Retrieval data.
        retrieval_targets(torch.Tensor): One-hot retrieval targets.
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

    # Evaluate
    # Out-of-sample extension
    train_code = U.sign()
    W = W_prime @ train_code

    # Generate query and retrieval code
    query_code = (query_data @ W).sign()
    retrieval_code = (retrieval_data @ W).sign()

    # Compute map
    mAP = mean_average_precision(
        query_code,
        retrieval_code,
        query_targets,
        retrieval_targets,
        topk,
    )

    return mAP
