import torch
from src.ocp_solver import define_ocp
from src.utils import load_model, load_config

def run_mpc_with_nn(config_path="config.yaml", nn_model=None):
    '''
    Run MPC using a neural network as the termninal cost.
    
    Args:
        config_path (str): Path to the configuration file
        nn_model (NeuralNetworkModel): Trained neural network model
        
    Returns:
        list: Results from MPC simulations
    '''
    
    # Load configuration
    config = load_config(config_path)
    
    # Load neural network model if not provided
    if nn_model is None:
        nn_model = torch.load(config["paths"]["model"])
        
    results = []
    ocp, x, u, _ = define_ocp(config)
    
    for test_state in config["mpc"]["test_states"]:
        ocp.set_initial_state(test_state)
        
        def terminal_cost(x):
            x_tensor = torch.tensor(x, dtype=torch.float32)
            return nn_model(x_tensor).item()
    
        ocp.set_terminal_cost(terminal_cost)
        
        solver = ocp.get_solver()
        solution = solver.solve()
        
        results.append({
            "trajectory": solution.get_state_trajectory(),
            "controls": solution.getr_control_trajectory (),
            "cost":solution.get_total_cost()
        })
        
    return results
        
        