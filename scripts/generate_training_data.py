import numpy as np
from src.ocp_solver import solve_ocp
from src.utils import save_data

def geneate_training_data(config):
    initial_states = []
    training_data = []
    
    for _ in range(config["ocp"]["num_initial_states"]):
        angle = np.random.uniform(
            config["ocp"]["iniitial_sates"]["angle"][0],
            config["ocp"]["initial_states"]["angle"][1]
        )
        angular_velocity = np.random.uniform(
            config["ocp"]["intial_states"]['angular_velocity'][0],
            config["ocp"]["intial_states"]['angular_velocity'][1]
        )
        initial_state = np.array([angle, angular_velocity])
        initial_states.append(initial_state)
        
        solution = solve_ocp(initial_state, config)
        training_data.append((initial_state, solution["cost"]))
        
        save_data(np.array(training_data), config["paths"]["training_data"])
        return training_data