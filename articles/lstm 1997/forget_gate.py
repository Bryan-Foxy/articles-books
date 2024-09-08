import torch

def forget_gate(W, X, h, b, activations_functions):
    """Forget gate of the LSTM"""
    array_data = torch.cat((h,b), axis = 0)
    f_t = activations_functions(torch.cat(W,array_data + b))
    return f_t