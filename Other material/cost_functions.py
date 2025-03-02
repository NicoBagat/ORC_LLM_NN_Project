# Cost functions used in the OCP including terminal cost that leverages the neural network's Value function

import torch

def compute_cost(x, u):
    # Define stage cost, e.g., penalizing state and control
    return torch.square(x).sum() + torch.square(u)

def terminal_csot(x, network):
    with torch.no_grad():
        return network(x).item()