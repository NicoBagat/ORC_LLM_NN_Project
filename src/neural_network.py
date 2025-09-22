import torch
import torch.nn as nn
from src.utils import load_config
from casadi import MX, Function
import l4casadi as l4c

class NeuralNetwork(nn.Module):
    '''
    Simple feedforward neural network to approximate the Value function
    Inputs:
        - Initial state x0
    Outputs:
        - Predicted optimal cost J(x0)
        
    input_size: # input layer neurons
    hidden_size: # hiddenlayer neurons
    output_size: # output layer neurons
    '''
    
    def __init__(self, nn_input_dim, nn_hidden_dim, nn_output_dim, activation=nn.Tanh(), ub=1):
        super().__init__()
        
        # Define the model architecture
        self.linear_stack = nn.Sequential(
            nn.Linear(nn_input_dim, nn_hidden_dim),
            activation,
            nn.Linear(nn_hidden_dim, nn_hidden_dim),
            activation,
            nn.Linear(nn_hidden_dim, nn_output_dim),
            activation,
        )
        self.ub = ub
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
    
    
    def create_casadi_function(self, config_path, load_weights=True):
        """
        Create a CasADi function from the neural network using l4casadi.
        Loads weights from the model path specified in the config if load_weights is True.
        """
        
        config = load_config(config_path)
        input_size = config["ocp"]["state_dim"]
        model_path = config["paths"]["model"]
        
        if load_weights:
            devide = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.load_state_dict(torch.load(model_path, map_location=devide))
            
        state = MX.sym("x", input_size)
        l4c_model = l4c.L4CasADi(self,
                                 device="cuda" if torch.cuda.is_available() else "cpu",
                                 name='nn_model',
                                 build_dir='models/l4c_build'
                                 )
        
        nn_model = l4c_model(state)
        
        # This is the function you can use in a CasADi problem
        nn_func = Function('nn_func', [state], [nn_model])
        return nn_func