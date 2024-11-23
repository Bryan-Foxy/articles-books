 
torch.manual_seed(42)

# Generate synthetic data: spiral trajectory
def generate_spiral_data(n_points=100):
    t = torch.linspace(0, 6.28, n_points)  # Time points (0 to 2*pi)
    x = t * torch.cos(t)  # x-coordinate
    y = t * torch.sin(t)  # y-coordinate
    return t, torch.stack([x, y], dim=1)

# Neural ODE model
class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 2)
        )

    def forward(self, t, y):
        return self.net(y)

# Training loop
def train_neural_ode(ode_func, t_train, trajectory, n_epochs=500, lr=0.01):
    optimizer = torch.optim.Adam(ode_func.parameters(), lr=lr)
    criterion = nn.MSELoss()
    loss_history = []

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # Neural ODE forward pass
        pred_trajectory = odeint(ode_func, trajectory[0], t_train)

        # Compute loss
        loss = criterion(pred_trajectory, trajectory)
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.6f}")
    
    return loss_history

# Visualize results
def plot_results(t_train, trajectory, pred_trajectory, loss_history):
    # Plot trajectory
    plt.figure(figsize=(12, 5))
    
    # Ground truth vs Predicted Trajectories
    plt.subplot(1, 2, 1)
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'g-', label='Ground Truth')
    plt.plot(pred_trajectory[:, 0].detach(), pred_trajectory[:, 1].detach(), 'r--', label='Predicted')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('Trajectory')

    # Loss history
    plt.subplot(1, 2, 2)
    plt.plot(loss_history, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss History')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Main script
if __name__ == "__main__":
    # Generate synthetic data
    t_train, trajectory = generate_spiral_data()

    # Define Neural ODE model
    ode_func = ODEFunc()

    # Train the model
    loss_history = train_neural_ode(ode_func, t_train, trajectory, n_epochs=500, lr=0.01)

    # Predict trajectory using the trained model
    pred_trajectory = odeint(ode_func, trajectory[0], t_train)

    # Visualize results
    plot_results(t_train, trajectory, pred_trajectory, loss_history)