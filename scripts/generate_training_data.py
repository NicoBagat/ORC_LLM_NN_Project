from src.utils import load_config
from src.dynamics import single_pendulum_dynamics, double_pendulum_dynamics
from src.ocp_solver import define_ocp, solve_ocp
from src.cost_functions import define_cost_function

def generate_training_data(config_path):
    config = load_config(config_path)
    
    # If 'dynamics' field is "single_pendulum" load single pendulum dynamics, else load double pendulum dynamics
    dynamics_fn = (
        single_pendulum_dynamics()
        if config["dynamics"] == "single_pendulum"
        else double_pendulum_dynamics
    )
    stage_cost, terminal_cost = define_cost_function(config)
    ocp = define_ocp(config, dynamics_fn, stage_cost, terminal_cost)
    
    training_data = []
    for x0 in config["initial_states"]:
        trajectory, controls, cost = solve_ocp(ocp, x0)
        training_data.append((x0, cost))
        
    return training_data