import yaml
import numpy as np

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