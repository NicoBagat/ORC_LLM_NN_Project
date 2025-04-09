import numpy as np
from src.utils import load_config
from src.dynamics import single_pendulum_dynamics, double_pendulum_dynamics
from src.ocp_solver import define_ocp, solve_ocp
from src.cost_functions import define_cost_function

def generate_training_data(config_path):
    config = load_config(config_path)
    
    # Generate random initial states
    angle_range = config["ocp"]["initial_state_ranges"]["angle"]
    angular_velocity_range = config["ocp"]["initial_state_ranges"]["angular_velocity"]
    num_states = config["ocp"]["num_initial_states"]
    
    initial_state = [
        [
            np.random.uniform(*angle_range),
            np.random.uniform(*angular_velocity_range),
        ]
        for _ in range(num_states)
    ]
    
    # Select dynamics based on configuration
    dynamics_fn = (
        single_pendulum_dynamics()
        if config["dynamics"] == "single_pendulum"
        else double_pendulum_dynamics
    )
    
    # Define the cost functions
    stage_cost, terminal_cost = define_cost_function(config)
    
    # Define the OCP
    ocp = define_ocp(config, dynamics_fn, stage_cost, terminal_cost)
    
    # Solve OCP for each initial state
    training_data = []
    for x0 in config["initial_states"]:
        trajectory, controls, cost = solve_ocp(ocp, x0)
        training_data.append((x0, cost))
        
    #Save training data to the specified path
    training_data_path = config["paths"]["training_data"]
    np.save(training_data_path, training_data)
    # If 'dynamics' field is "single_pendulum" load single pendulum dynamics, else load double pendulum dynamics


    

        
    return training_data