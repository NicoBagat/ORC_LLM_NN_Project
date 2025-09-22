from scripts.generate_training_data import generate_training_data
from scripts.train_nn import train_neural_network
from scripts.run_mpc_with_nn import run_mpc_with_nn
from src.utils import load_config, load_data

def test_generate_training_data():
    config = load_config("config.yaml")
    data = generate_training_data(config)
    print(f"Generated training data: {len(data)} samples")
    print("Sample: ", data[0] if data else "No data generated")
    
def test_train_neural_network():
    config = load_config("config.yaml")
    training_data = load_data(config["paths"]["training_data"])
    nn_model = train_neural_network(training_data, config)
    print("Neural network trained and exported to ONNX.")
    
def test_run_mpc_with_nn():
    results = run_mpc_with_nn("config.yaml")
    print(f"MPC with NN terminal cost ran for {len(results)} test states.")
    for i, res in enumerate(results):
        print(f"Test {i+1}: Cost={res['cost']:.3f}, Terminal Cost={res['terminal_cost']:.3f}")
        
if __name__ == "__main__":
    print("Testing data generation...")
    test_generate_training_data()
    print("\nTesting neural network training...")
    test_train_neural_network()
    print("\nTesting MPC with neural network terminal cost...")
    test_run_mpc_with_nn()