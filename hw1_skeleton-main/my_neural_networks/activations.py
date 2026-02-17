import numpy as np
import torch

EPSILON = 1e-14


def cross_entropy(X, y_1hot, epsilon=EPSILON):
    """Cross Entropy Loss

        Cross Entropy Loss that assumes the input
        X is post-softmax, so this function only
        does negative loglikelihood. EPSILON is applied
        while calculating log.

    Args:
        X: (n_neurons, n_examples). softmax outputs
        y_1hot: (n_classes, n_examples). 1-hot-encoded labels

    Returns:
        a float number of Cross Entropy Loss (averaged)
    """
    E = - (1/ X.shape[1]) * torch.sum(y_1hot *(X - torch.log(torch.sum(torch.exp(X)))) )
    return E
    # raise NotImplementedError


def softmax(X):
    """Softmax

        Regular Softmax

    Args:
        X: (n_neurons, n_examples). 

    Returns:
        (n_neurons, n_examples). probabilities
    """
    X_exp = torch.exp(X) 

    return X_exp / torch.sum (X_exp,dim = 1, keepdim = True  )
    # raise NotImplementedError


def stable_softmax(X):
    """Softmax

        Numerically stable Softmax

    Args:
        X: (n_neurons, n_examples). 

    Returns:
        (n_neurons, n_examples). probabilities
    """

    X_exp = torch.exp(X-torch.max(X,dim=1,keepdim=True))
    
    return X_exp / torch.sum(X_exp,dim=1, keepdim=True)
    # raise NotImplementedError


def relu(X):
    """Rectified Linear Unit

        Calculate ReLU

    Args:
        X: torch.Tensor

    Returns:
        A torch.Tenor whereThe shape is the same as X but clamped on 0
    """

    return torch.clamp(X, min=0.0)

    # raise NotImplementedError


def sigmoid(X):
    """Sigmoid Function

        Calculate Sigmoid

    Args:
        X: torch.Tensor

    Returns:
        A torch.Tensor where each element is the sigmoid of the X.
    """
    X_nexp = torch.exp(-X)
    return 1.0 / (1 + X_nexp)
