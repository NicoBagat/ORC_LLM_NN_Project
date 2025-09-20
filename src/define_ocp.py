import casadi as cs
from src.dynamics import single_pendulum_dynamics, double_pendulum_dynamics

def define_ocp(config):
    ''' 
    Define the OCP based on the configuration 
        
        Args:
        config (dict): Configuration dictionary
            
        Returns:
        tuple: A tuple containing OCP problem parameters, CasADi symbolic variables and dynamics
    '''

    # Load "ocp" parameters from config file
    state_dim = config["ocp"]["state_dim"]
    control_dim = config["ocp"]["control_dim"]
    dt = config["ocp"]["dt"]
    cost_weights = config["ocp"]["cost_weights"]
    
    # Define symbolic variables
    x = cs.MX.sym("x", state_dim)   # State variable
    u = cs.MX.sym("u", control_dim) # Control variable
    
    # SELECT DYNAMICS FUNCTION
    if config["ocp"]["dynamics"] == "single_pendulum":
        dynamics_fn = single_pendulum_dynamics(config)
    else:
        dynamics_fn = double_pendulum_dynamics(config)
    
    # Define stage cost function
    def stage_cost(x, u):
        return cost_weights[0] * cs.sumsqr(x) + cost_weights[1] * cs.sumsqr(u)
    
    return {
        "cs" : cs,
        "x": x,
        "u": u,
        "dynamics_fn": dynamics_fn,
        "stage_cost": stage_cost,
        "dt": dt
    }, x, u, dynamics_fn
    