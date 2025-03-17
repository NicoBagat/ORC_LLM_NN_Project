import yaml
from scripts.generate_training_data import generate_training_data
from scripts.train_neural_network import train_neural_network
from scripts.run_mpc_with_nn import run_mcp_with_nn

def load_config(config_path = "config.yamnl"):
    """ Load the project specific configuration file """
    with open(config_path, "r") as file:
        return yaml.safe_load(file)
    

# MAIN FUNCTION
""" Main function used to execute the the various project's steps """
def main():
    config = load_config()
    
    # STEP 1: Generate training data
    if config["Steps"]["generate_training_data"]:
        print("[INFO] STEP 1: Generating training data (solving OCPs traditionally)...")
        generate_training_data(config)
        
    # STEP 2: Train neural network
    if config["Steps"]["train_neural_network"]:
        print("[INFO] STEP 2: Training the neural network ...")
        train_neural_network(config)
        
    # STEP 3: Run MPC with the neural network as terminal cost
    if config["Steps"]["run_mpc_with_nn"]:
        print["Steps"]["STEP 3:  Running MPC with neural network terminal cost ..."]
        run_mcp_with_nn(config)
        
if __name__ == "__main__":
    main()