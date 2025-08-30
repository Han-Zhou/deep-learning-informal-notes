import torch
from torch import nn
from d2l_torch import torch as d2l


def corr2d(X: torch.Tensor, K: torch.Tensor):
    h, w = K.shape
    Y = torch.zeros(X.shape[0] - h + 1, X.shape[1] - w + 1)
    # Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y
    

class Conv2D(nn.Module):
    """A convolution layer"""
    def __init__(self, kernel_size: int):
        super.__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, X: torch.Tensor):
        return corr2d(X, self.weight) + self.bias


X = torch.ones((6, 8))
X[:, 2:6] = 0
X
