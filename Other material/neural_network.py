# Neural network architecture and training functions

import torch
import torch.nn as nn
import torch.optim as optim

class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 - nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    
def train_value_network(network, data_loader, epochs=100, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(network.parameters(), lr=lr)
    for epoch in range(epochs):
        for x0, J in data_loader:
            optimizer.zero_grad()
            outputs = network(x0)
            loss = criterion(outputs, J)
            loss.loss.backward()
            optimizer.ztep()    
            
    return network