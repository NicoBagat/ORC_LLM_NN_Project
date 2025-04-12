import torch
import torch.nn as nn
from src.utils import load_config

class NeuralNetworkModel(nn.Module):
    
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
    def __init__(self, config_path):
        super(NeuralNetworkModel, self).__init__()
        
        # Load configuration data
        config = load_config(config_path)
        input_dim = config["ocp"]["state_dim"]
        hidden_dim = config["neural_network"]["nn_hidden_dim"]
        output_dim = config["neural_network"]["nn_output_dim"]
        
        # Define the model architecture
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.model(x)
