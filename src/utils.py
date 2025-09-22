import os
import yaml
import numpy as np
import torch

def load_config(config_path = "config.yaml"): 
    """
    Load the configuration file
    
    Args:
        config_path(str): Path to the config.yaml file
        
    Returns():
        dict: Parsed configuration dictionary
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file) # Parse the YAML file
    return config

def save_data(data, path):
    """
    Save numpy array data to a specified path
    

    Args:
        data (numpy.ndarray): data to save
        path (str): File path to save the data
        """
    os.makedirs(os.path.dirname(path), exist_ok=True) # Create directory if it doesn't exist
    np.save(path, data) # Save the data
    print(f"Data saved to {path}") 
    
def load_data(path):
    """
    Load numpy array data from a specified path.
    
    Args:
        path (str): File apth to load the data from
    
    Returns:
        numpy.ndarray: Loaded data
    """
    return np.load(path, allow_pickle=True) # Load the data with allow_pickle=True to handle object arrays

def save_model(model, path): 
    """
    Sve a PyTorch model to the specified path.
    
    Args:
        model (torch.nnModule): PyTorch model to save
        path (str): File path to save the model
    """
    os.makedirs(os.path.dirname(path), exist_ok=True) # Create directory if it doesn't exist
    torch.save(model.state_dict(), path) # Save only the model parameters
    print(f"Model saved to {path}")
    
def load_model(path, model_class, config):
    """
    Load a PyTorch model from the specified path.
    
    Args:
        path (str): File path to load the model from
        model_class (torch.nn.Module)
        config (dict): Configuration dictionary
        
    Returns:
        torch.nn.Module: Loaded PyTorch model.
    """
    
    model = model_class(config) # Initialize the model architecture
    model.load_state_dict(torch.load(path)) # Load the model parameters
    print(f"Model loaded from {path}") 
    return model