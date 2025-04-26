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
        config = yaml.safe_load(file)
    return config

def save_data(data, path):
    """
    Save numpy array data to a specified path
    

    Args:
        data (numpy.ndarray): data to save
        path (str): File path to save the data
        """
    np.save(path, data)
    print(f"Data saved to {path}")
    
def load_data(path):
    """
    Load numpy array data from a specified path.
    
    Args:
        path (str): File apth to load the data from
    
    Returns:
        numpy.ndarray: Loaded data
    """
    return np.load(path)

def save_model(model, path):
    """
    Sve a PyTorch model to the specified path.
    
    Args:
        model (torch.nnModule): PyTorch model to save
        path (str): File path to save the model
    """
    torch.save(model.state_dict(), path)
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
    
    model = model_class(config)
    model.load_state_dict(torch.load(path))
    print(f"Model loaded from {path}")
    return model

def normalize_data(x, y):
    """
    Normalize input and output data for training.
    
    Args:
        x (numpy.ndarray): Input features.
        y (numpy/ndarray): Ouput targets.
        
        Returns:
        tuple: Normalized input and output data.
        
    """
    x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
    y_mean, y_std = np.mean(y, axis=0), np.std(y, axis=0)
    
    x_norm = (x - x_mean) / x_std
    y_norm = (y - y_mean) / y_std
    
    return x_norm, y_norm
