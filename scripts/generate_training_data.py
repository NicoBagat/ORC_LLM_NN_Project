import numpy as np
from src.utils import load_config, save_data
from src.ocp_solver import define_ocp, solve_ocp
from src.dynamics import single_pendulum_dynamics, double_pendulum_dynamics
from src.cost_functions import define_cost_function

def generate_training_data(config):
    """
    Generate training data by solving OCPs with different initial states
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        List[Tuple[np.ndarray, float]]: Training data containing initial states and corresponding optimal costs
    """
    
    #Load configuration 
    config = load_config(config)
    
    # Select dynamics function absed on configuration
    dynamics_fn = (
        single_pendulum_dynamics(config)
        if config["ocp"]["dynamics"] == "single_pendulum"
        else double_pendulum_dynamics(config)
    )
    
    # Generate initial states
    num_initial_states = config["ocp"["num_initial_states"]["angle"]]
    initial_states = []
    
    angular_velocity = np.random.uniform(
        *config["ocp"]["initial_states"]["angular_velocity"]
    )
    
    for _ in range(num_initial_states):
        angle = np.random.uniform(*config["ocp"]["initial_states"]["angle"])
        angular_velocity = np.random.uniform(*config["ocp"]["initial_states"]["angular_velocity"])
        initial_states.append([angle, angular_velocity])
        
    # Define the stage and terminal costs
    stage_cost, terminal_cost = define_cost_function(config)
    
    # Define OCP
    ocp = define_ocp(config, dynamics_fn, stage_cost, terminal_cost)
    
    # Solve OCP for each initial state and collect training data
    training_data = []
    
    for x0 in initial_states:
        _, _, optimal_cost = solve_ocp(ocp, x0)
        training_data.append((x0, optimal_cost))
        
    return training_data

if __name__ =="__main__":
    #Pathg to configuration file
    config_path = "config.yaml"
    training_data = generate_training_data(config_path)
    print(f"Generated training data: {len(training_data)} data points.")