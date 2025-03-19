import torch.nn as nn

class ValueFunctionNN(nn.Module):
    
    '''
    input_dim: # input layer neurons
    hidden_dim: # hiddenlayer neurons
    output_dim: # output layer neurons
    '''
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ValueFunctionNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.model(x)