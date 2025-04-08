from src.ocp_solver import define_ocp, solve_ocp
from src.dynamics import single_pendulum_dynamics, double_pendulum_dynamics

def run_mpc_with_nn(config, nn_model):
    dynamics_fn = (
        single_pendulum_dynamics()
        if config["dynamics"] == "single_pendulum"
        else double_pendulum_dynamics()
    )
    ocp = define_ocp(config, dynamics_fn, stage_cost = None, terminal_cost = nn_model)
    
    results = []
    for x0 in config["mpc_initial_states"]:
        trajectory, controls, cost = solve_ocp(ocp, x0)
        results.append((trajectory, controls, cost))
        
    return results