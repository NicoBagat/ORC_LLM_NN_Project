import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset
from src.utils import load_config
from src.neural_network import NeuralNetworkModel

def train_neural_network(training_data, config):
    
    # Load configuration
    config = load_config(config)
    
    # Extract neural network configuration
    input_dim = config["ocp"]["state_dim"]
    hidden_dim = config["neural_network"]["nn_hiddent_dim"]
    output_dim = config["neural_network"]["nn_output_dim"]
    epochs = config["neural_network"]["nn_epochs"]
    batch_size = config["neural_network"]["nn_batch_size"]
    learning_rate = config["neural_network"]["nn_learning_rate"]
    
    # Load training data
    training_data_path = config["paths"]["training_data"]
    training_data = torch.tensor(torch.load(training_data_path))
    x_train, y_train = training_data[:, :-1], training_data[:, -1].unsqueeze(1)
    
    # Create DataLoader
    dataset = TensorDataset(x_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Define the model
    model = NeuralNetworkModel(input_dim, hidden_dim, output_dim)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # TRAINING LOOP
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in dataloader:
            # Forward pass
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        # Print epoch details
        print(f"Epoch {epoch +1}/{epochs}, Loss: {epoch_loss:.4f}")
        
        # Save the trained model
        model_path = config["paths"]["model"]
        torch.save(model.state_dict(), model_path)
        print(f"Model savedd to {model_path}")

if __name__ == "__main__":
    
    # Path to config.yaml
    config_path = "config.yaml"
    train_neural_network(config_path)
