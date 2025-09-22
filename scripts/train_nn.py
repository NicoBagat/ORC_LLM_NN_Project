import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.neural_network import NeuralNetwork
from src.utils import save_model, load_data, load_config

def train_neural_network(config_path="config.yaml"):
    '''
    Train a neural network on the generated training data.
    Args:
        config_path (str): Path to the configuration file.
    Returns:
        NeuralNetworkModel: Trained neural network model.
    '''
    # Load configuration
    config = load_config(config_path)
    
    # Load trainig data
    training_data = load_data(config["paths"]["training_data"])
    x_data, y_data = zip(*training_data)
    x_data, y_data = np.array(x_data), np.array(y_data)
    
    # Normalize data
    x_mean = np.mean(x_data, axis=0)    #'x' mean
    x_std = np.std(x_data, axis=0)      #'x' standard deviation
    x_std[x_std ==0] = 1               # Prevent division by zero
    
    y_mean = np.mean(y_data)        #'y' mean
    y_std = np.std(y_data)          #'y' standard deviation
    if y_std ==0:
        y_std = 1                     # Prevent division by zero
        
    x_data = (x_data - x_mean) / x_std  # Normalize 'x' data
    y_data = (y_data - y_mean) / y_std # Normalize 'y' data
    
    # Prepare the Dataloader
    dataset = TensorDataset(torch.tensor(x_data, dtype=torch.float32),
                            torch.tensor(y_data, dtype=torch.float32))
    
    dataloader = DataLoader(dataset, batch_size=config["neural_network"]["nn_batch_size"], shuffle=True)
    
    # INITIALIZE THE NEURAL NETWORK
    input_dim = config["ocp"]["state_dim"]
    nn_hidden_dim = config["neural_network"]["nn_hidden_dim"]
    nn_output_dim = config["neural_network"]["nn_output_dim"]
    nn_model = NeuralNetwork(input_dim, nn_hidden_dim, nn_output_dim)
    optimizer = torch.optim.Adam(nn_model.parameters(), lr=config["neural_network"]["nn_learning_rate"])
    loss_fn = torch.nn.MSELoss() #Criterion for MSE (Mean Square Error) measurement
    
    # Train the neural network
    for epoch in range(config["neural_network"]["nn_epochs"]):
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()   #Reset the gradients for all optimized class
            predictions = nn_model(x_batch) #Forward pass: compute predicted y by passing x to the model
            loss = loss_fn(predictions, y_batch) #Compute the loss
            loss.backward() 
            
            '''
            backward(): method used in PyTorch to calculate the gradient dureing the backward pass in the neural network.

                -> not calling backward will cause the gradients for the tensor to not be computed
            '''
            optimizer.step() #Perform optimization step
    
    # Save the trained model
    save_model(nn_model, config["paths"]["model"])
    print("Neural network training complete.")
    return nn_model
    
    
    
