import torch

def sigmoid(x):
    return 1/(1+torch.exp(-x))

def tanh(x):
    return (torch.exp(x) - torch.exp(-x))/(torch.exp(x) + torch.exp(-x))
