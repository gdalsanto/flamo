import torch
from torch import nn
from torch.optim import LBFGS

# TODO Add documentation
# Define the loss function
class MLS(nn.Module):
    def __init__(self, G, target_interp):
        super().__init__()
        self.G = G
        self.target_interp = target_interp

    def forward(self, x):
        return torch.mean(torch.pow(torch.matmul(self.G, x) - self.target_interp, 2))


def minimize_LBFGS(G, target_interp, lower_bound, upper_bound, num_freq, max_iter=100):

    initial_guess = nn.Parameter(torch.ones(num_freq + 1))
    assert len(lower_bound) == len(upper_bound) == len(initial_guess), 'The number of bounds must be equal to the number of gains.'
    
    # Create an instance of LBFGS optimizer
    optimizer = LBFGS([initial_guess])
    criterion = MLS(G, target_interp)
    # Define a closure for the LBFGS optimizer
    def closure():
        optimizer.zero_grad()
        loss = criterion(initial_guess)
        loss.backward()
        initial_guess.data.clamp_(lower_bound, upper_bound) 
        return loss
    
    # Perform optimization
    for i in range(max_iter):
        optimizer.step(closure)

    # Get the optimized result
    return initial_guess
