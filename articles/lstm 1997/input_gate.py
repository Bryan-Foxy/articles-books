import torch
import forget_gate as f_t
from activations_functions import sigmoid, tanh

def input_gate(W, X, h, b, c_t):
    i_t = sigmoid(torch.dot(W,torch.cat((h, X), axis = 0)) + b)
    C_tilde = tanh(torch.dot(W,torch.cat((h, X), axis = 0)) + b)
    C_t = (f_t * c_t) + (i_t * C_tilde)
    return C_t