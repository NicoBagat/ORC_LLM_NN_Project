import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from src.utils import load_config, save_model, normalize_data
from src.neural_network import NeuralNetworkModel

def train_neural_network(training_data, config):
    
    # Normalize data
    x_train, y_train = normalize_data(training_data[:, :-1], training_data[:, -1].reshape(-1, 1))
    
    # Create dataset and split into train/validation
    dataset = TensorDataset(torch.tensor(x_train).float(), torch.tensor(y_train).float())
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create Dataloaders
    batch_size = config["neural_network"]["nn_batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = NeuralNetworkModel(config)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["neural_network"]["nn_learning_rate"])
    
    # Training loop with early stopping
    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0
    
    for epoch in range(config["neural_network"]["nn_epochs"]):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad
        
if __name__ == "__main__":
    
    # Path to config.yaml
    config_path = "config.yaml"
    train_neural_network(config_path)
