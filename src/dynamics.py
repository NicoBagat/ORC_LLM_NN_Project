import l4casadi as l4c
from src.utils import load_config

def single_pendulum_dynamics(config):    
    ''' 
        Define the dynamics for a single pendulum using config dimensions and parameters
        Args:
            config (dict): Configuration dictionary loaded from config.yaml
        
        Returns:
            Callable: l4casadi-compatible dynamics function
        Symbolic dynamics of a single pendulum using l4casadi
    '''
    state_dim = config["state_dim"]
    control_dim = config["control_dim"]
    
    # Symbolic definition of state and control variables
    x = l4c.SX.sym("x", state_dim)
    u = l4c.SX.sym("u", control_dim)
    
    theta, omega = x[0], x[1]
    torque = u[0]
    g, L, m = 9.81, 1.0, 1.0
    
    theta_dot = omega
    omega_dot = (-m * g * L * l4c.sin(theta) + torque) / ( m * L**2)
    
    dxdt = l4c.vertcat(theta_dot, omega_dot)
    dynamics_fn = l4c.Function("single_pendulum_dyunamics", [x, u], [dxdt])

    return dynamics_fn

def double_pendulum_dynamics(config):
    """
    Define the dynamics for a double pendulum.
    Args:
        config (dict): Configuration dictionary loaded from config.yaml
    Returns:
        Callable: l4casadi-compatible dynamics function.
    """
    state_dim = config["state_dim"]
    control_dim = config["control_dim"]
    
    x = l4c.SX.sym("x", state_dim)
    u = l4c.SX.sym("u", control_dim)

    # Placeholder dynamics for the double pendulum
    # Replace this with the symbolic formulation for the double pendulum
    
    dxdt = l4c.vertcat(*[0] * state_dim) # Replace with actual dynamics

    dynamics_fn = l4c.Function("double_pendulum_dynamics", [x, u], [dxdt])
    return dynamics_fn