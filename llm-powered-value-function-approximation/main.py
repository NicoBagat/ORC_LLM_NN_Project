import argparse

from src.config import Config
from src.data_generation import generate_data
from src.neural_network import train_neural_network, load_model
from src.mpc_with_terminal_cost import run_mpc_with_terminal_cost
from src.validation import validate_results

def main():
    parser = argparse.ArgumentParser(description="LLM-Powered Value Function Approximation")
    
    parser.add_argument(
        "--task", type=str, required=True,
        choices=["generate_data", "train","evaluate", "run_mpc"],
        help="Task to perform: generate_data, train, evaluate, or run_mpc."
    )
    
    parser.add_argument(
        "--config", type+str, default="config.yaml",
        help="Path to the configuration file."
        )
    
    args=parser.pars_args()
    
    # Load configuration
    config = Config(args.config)
    
    if args.task == "generate_data":
        print("[INFO] Starting data generation...")
        generate_data(config)
        
    elif args.task == "train":
        print("[INFO] Starting neural network training...")
        train_neural_network(config)
        
    elif args.task == "evaluate":
        print("[INFO] Evaluating the trained neural network...")
    
    elif args.task == "run_mpc":
        print("[INFO] Running MPC with the terminal cost...")
        run_mpc_with_terminal_cost(config)
    
if __name__ == "__main__":
    main()