from scripts.generate_training_data import generate_training_data
from scripts.train_nn import train_neural_network
from scripts.run_mpc_with_nn import run_mpc_with_nn
from src.utils import load_config
from scripts.visualize_training_data import visualize_training_data

def main():
    # Load configuration
    config = load_config("config.yaml")
    
    # Step 1 : Generate training data
    print("[STEP_1]: Generating training data ...")
    training_data = generate_training_data(config)
    print(f"Generated {len(training_data)} data points.")
    
    # Step 2: Visualize training data
    print("[STEP_2]: Visualizing training data ...")
    visualize_training_data(config)
    print("Training data visualization complete.")
    
    # Step 3: Train neural network
    print("[STEP_3]: Training neural network ...")
    nn_model = train_neural_network("config.yaml")
    print("Neural network training complete.")
    
    # Step 4: Run MPC with trained neural network
    print("[STEP_4]: Running MPC with neural network as terminal cost ...")
    mpc_results = run_mpc_with_nn(config, nn_model)
    print("MPC simulations complete.")
    
    # Output results (example)
    print("MPC Results:")
    for i, result in enumerate(mpc_results):
        print(f"Simulation {i+1}: Cost = {result['']}")

if __name__ == "__main__":
    main()