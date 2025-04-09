from scripts.generate_training_data import generate_training_data
from scripts.train_nn import train_neural_network
from scripts.run_mpc_with_nn import run_mpc_with_nn
from src.utils import load_config

def main():
    # Load configuration
    config = load_config("config.yaml")
    
    # Step 1 : Generate training data
    print("[STEP_1]: Generating trainind data ...")
    training_data = generate_training_data(config)
    print(f"Generated {len(training_data)} data points.")
    
    # Step 2: Train neural network
    print("[STEP_2]: Training neural network ...")
    nn_model = train_neural_network(training_data, config)
    print("Nerual network training complete.")
    
    # Step 3: Run MPC with trained neural network
    print("[STEP_3]: Running MPC with neural network as terminal cost ...")
    mpc_results = run_mpc_with_nn(config, nn_model)
    print("MPC simulationa complete.")
    
    # Output results (example)
    print("MPC Results:")
    for i, result in enumerate(mpc_results):
        trajectory, controls, cost = result
        print(f"Simulation {i+1}: Cost = {cost}")

if __name__ == "__main__":
    main()