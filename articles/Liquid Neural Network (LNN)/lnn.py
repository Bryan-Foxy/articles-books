import torch

class LiquidNeuralState(torch.nn.Module):
    """
    Liquid Neural State (LNS) layer for processing sequences of embeddings.
    This layer computes the liquid state based on the input sequence and a set of parameters.
    """
    def __init__(self, input_dim, hidden_dim):
        super(LiquidNeuralState, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.W_in = torch.nn.Linear(self.input_dim, self.hidden_dim)
        self.W_h = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.tau = torch.nn.Parameter(torch.ones(hidden_dim))
    
    def forward(self, x, h):
        """
        Forward pass for the Liquid Neural State layer.
        We first compute the input and hidden states using the linear layers defined in __init__.
        We then apply the liquid state equation to compute the new hidden state.
        Args:
            x (torch.Tensor): The input sequence of shape (B, T, input_dim).
            h (torch.Tensor): The previous hidden state of shape (B, hidden_dim).
        Returns:
            h_new (torch.Tensor): The new hidden state of shape (B, hidden_dim).
        """
        dx = torch.tanh(self.W_in(x) + self.W_h(h))
        h_new = h + (dx - h) / self.tau
        return h_new

class LiquidNeuralNetwork(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LiquidNeuralNetwork, self).__init__()
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.layers.append(LiquidNeuralState(in_dim, hidden_dim))
        self.fc_out = torch.nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        B, seq_len, _ = x.shape
        h = [torch.zeros(B, layer.hidden_dim, device = x.device) for layer in self.layers]
        for t in range(seq_len):
            x_t = x[:,t,:]
            for i, layer in enumerate(self.layers):
                h[i] = layer(x_t, h[i])
                x_t = h[i]
        out = self.fc_out(h[-1])
        return out


