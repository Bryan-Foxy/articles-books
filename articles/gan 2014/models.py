"""
Author: FOZAME ENDEZOUMOU Armand Bryan
"""
import torch

class Gen(torch.nn.Module):
    """Build the Generator for the GANs. This part of the framework will generate synthetic images"""
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.main = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size*2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size*2, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size),
            torch.nn.Tanh()
        )

    def forward(self,x):
        return self.main(x)
    

class Disc(torch.nn.Module):
    """Build the Discriminator for the GANs. this part of the framework will classify images send by the Generator I will answer if it's a fake or not"""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.main = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(hidden_size, hidden_size//2),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(hidden_size//2, 1),
            torch.nn.Sigmoid()

        )

    def forward(self,x):
        return self.main(x)

