# Author: FOZAME ENDEZOUMOU Armand Bryan



# Neural ODEs are a type of neural network that models continuous-time data using differential equations. 
# Instead of using a discrete sequence of time as in traditional recurrent networks (RNNs), they follow continuous dynamics. 
# The model is based on the solution of a differential equation that governs the evolution of the state x(t) over time, 
# and this evolution is driven by a neural network that takes into account the current state x(t) , an input I(t) , time t , 
# and a set of parameters \theta.

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt 
from tqdm import tqdm
from torchdiffeq import odeint, odeint_adjoint, odeint_event

class ODE(nn.Module):
	def __init__(self, theta):
		super(ODE, self).__init__()
		self.theta = nn.Parameter(theta)

	def forward(self, t, z):
		return self.theta * z

def loss(z_t1):
	target = torch.tensor([1.0])
	return (z_t1 - target).pow(2).mean()		

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
	# Hyperparams
	t0 = 0.0
	t1 = 1.0
	EPOCHS = 100
	z_t0 = torch.tensor([0.5])
	theta_init = torch.tensor([0.1], requires_grad = True)

	# Model
	loss_values = []
	theta_values = []
	ode_func = ODE(theta = theta_init)
	print(f"Number of parameters: {count_parameters(ode_func)}")
	optimizer = optim.SGD(ode_func.parameters(), lr = 1e-3)

	for epoch in tqdm(range(EPOCHS)):
		optimizer.zero_grad()
		z_t1 = odeint(ode_func, 
			z_t0,
			torch.tensor([t0, t1]))[-1]
		loss_val = loss(z_t1)
		loss_val.backward()
		optimizer.step()
		loss_values.append(loss_val.item())
		theta_values.append(ode_func.theta.item())
		print(f"Epoch {epoch + 1} | Loss: {loss_val.item():.4f} | Theta: {ode_func.theta.item():.4f}")

	# Plot Loss
	plt.subplot(1, 2, 1)
	plt.plot(loss_values, label = "Loss", color = "blue")
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.title("Evolution of the loss")
	plt.legend()
	plt.grid(True)

	plt.subplot(1,2,2)
	plt.plot(theta_values, label = "theta", color = "red")
	plt.xlabel("Epoch")
	plt.ylabel("Theta")
	plt.title("Evolution of Theta")
	plt.grid()
	plt.tight_layout()
	plt.show()




