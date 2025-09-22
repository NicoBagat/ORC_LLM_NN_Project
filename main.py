from scripts.generate_training_data import generate_training_data
from scripts.train_nn import train_neural_network
from scripts.run_mpc_with_nn import run_mpc_with_nn
from src.utils import load_config
from scripts.visualize_training_data import visualize_training_data

def main():
    # Load configuration
    config = load_config("config.yaml")
    
    # Step 1 : Generate training data
    print(f"\n-----------------------------------------------\n")
    print("[STEP_1]: Generating training data ...")
    print(f"\n-----------------------------------------------\n")
    training_data = generate_training_data(config)
    print(f"Generated {len(training_data)} data points.")
    
    # Step 2: Visualize training data
    print(f"\n-----------------------------------------------\n")
    print("[STEP_2]: Visualizing training data ...")
    print(f"\n-----------------------------------------------\n")
    visualize_training_data(config)
    print("Training data visualization complete.")
    
    # Step 3: Train neural network
    print(f"\n-----------------------------------------------\n")
    print("[STEP_3]: Training neural network ...") 
    print(f"\n-----------------------------------------------\n")
    nn_model = train_neural_network("config.yaml")
    print("Neural network training complete.")
    
    # Step 4: Run MPC with trained neural network
    print(f"\n-----------------------------------------------\n")
    print("[STEP_4]: Running MPC with neural network as terminal cost ...")
    print(f"\n-----------------------------------------------\n")
    mpc_results = run_mpc_with_nn("config.yaml")
    print("MPC simulations complete.")
    
    # Output results (example)
    print(f"\n-----------------------------------------------\n")
    print("MPC Results:")
    print(f"\n-----------------------------------------------\n")
    for i, result in enumerate(mpc_results):
        print(f"Simulation {i+1}: Cost = {result['cost']}, Terminal Cost = {result.get('terminal_cost', 'N/A')}")

if __name__ == "__main__":
    main()