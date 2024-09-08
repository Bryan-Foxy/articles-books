import torch
from activations_functions import sigmoid, tanh

def output_gate(W, X, h, b, C_t):
    i_o = sigmoid(torch.dot(W,torch.cat((h,X), axis = 0)) + b)
    h_t = i_o * tanh(C_t)
    return h_t