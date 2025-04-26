import torch
from LNN import LiquidNeuralNetwork

def count_parameters(model):
    """
    Count the number of parameters in a PyTorch model.
    Args:
        model (torch.nn.Module): The PyTorch model.
    Returns:
        int: The number of parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    input_dim = 10
    hidden_dim = 20
    num_layers = 2
    output_dim = 5

    model = LiquidNeuralNetwork(input_dim, hidden_dim, num_layers, output_dim)
    print(f"Number of parameters in the model: {count_parameters(model)}")

    x = torch.randn(32, 50, input_dim)
    output = model(x)
    print(f"Output shape: {output.shape}")