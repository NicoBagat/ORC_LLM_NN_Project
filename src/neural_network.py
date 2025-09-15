import torch
import torch.nn as nn
from src.utils import load_config

class NeuralNetwork(nn.Module):
    '''
    Neural network to approximate the Value function
    Inputs:
        - Initial state x0
    Outputs:
        - Predicted optimal cost J(x0)
        
    input_dim: # input layer neurons
    hidden_dim: # hiddenlayer neurons
    output_dim: # output layer neurons
    '''
    
    def __init__(self, input_size, hidden_size, output_size, activation=nn.Tanh(), ub=None):
        super().__init__()
        
        # Define the model architecture
        self.linear_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            activation,
            nn.Linear(hidden_size, hidden_size),
            activation,
            nn.Linear(hidden_size, output_size),
            activation,
        )
        self.ub = ub if ub is not None else 1
        self.initialize_weights()
        
    def forward(self, x):
        out = self.linear_stack(x) * self.ub
        return out
    
    def initialize_weights(self):
        for layer in self.linear_stack:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    @classmethod
    def from_config(cls, config_path):
        config = load_config(config_path)
        input_size = config["ocp"]["state_dim"]
        hidden_size = config["neural_network"]["nn_hidden_dim"]
        output_size = config["neural_network"]["nn_output_dim"]
        activation = nn.Tanh() # or nn.ReLU() if you prefer
        ub = config["neural_network"].get("nn_output_ub", 1)
        
        return cls(input_size, hidden_size, output_size, activation, ub)