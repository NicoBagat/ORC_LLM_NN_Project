import l4casadi as l4c

def single_pendulum_dynamics(config):    
    ''' 
        Define the dynamics for a single pendulum using config dimensions and parameters
        Args:
            config (dict): Configuration dictionary loaded from config.yaml
        
        Returns:
            Callable: l4casadi-compatible dynamics function
        Symbolic dynamics of a single pendulum using l4casadi
    '''
    
    # Load parameters from configuration file
    state_dim = config["state_dim"]
    control_dim = config["control_dim"]
    g = config["parameters"]["gravity"]
    L = config["parameters"]["length"]
    m = config["parameters"]["mass"]
    
    # Symbolic definition of state and control variables
    x = l4c.SX.sym("x", state_dim)
    u = l4c.SX.sym("u", control_dim)

    theta, omega = x[0], x[1]
    torque = u[0]
    
    # Define the dynamics
    theta_dot = omega
    omega_dot = (-m * g * L * l4c.sin(theta) + torque) / (m * L**2)
    
    dxdt = l4c.vertcat(theta_dot, omega_dot)
    
    # Create and return the dynamics function
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
    # Load parameters from configuration file
    state_dim = config["state_dim"]
    control_dim = config["control_dim"]
    g = config["parameters"]["gravity"]
    L1 = config["parameters"]["length_1"]
    L2 = config["parameters"]["length_2"]
    m1 = config["parameters"]["mass_1"]
    m2 = config["parameters"]["mass_2"]
    
    # Define the state and control variables
    x = l4c.SX.sym("x", state_dim)
    u = l4c.SX.sym("u", control_dim)

    # Define the dynamics
    theta1, omega1, theta2, omega2 = x[0], x[1], x[2], x[3]
    torque1, torque2 = u[0], u[1]
    
    # (angular) Speed
    theta1_dot = omega1
    theta2_dot = omega2
    
    # (angular) Acceleration 
    omega1_dot = (m2 * g * l4c.sin(theta2) - m1 * g * L1 * l4c.sin(theta1) + torque1) / (m1 * L1**2)
    omega2_dot = (m2 * g * l4c.sin(theta2) - m2 * L2 * l4c.sin(theta1) + torque2) / (m2 * L2**2)
    
    dxdt = l4c.vertcat(theta1_dot, omega1_dot, theta2_dot, omega2_dot)


    # Create and return the dynamics funciton
    dynamics_fn = l4c.Function("double_pendulum_dynamics", [x, u], [dxdt])
    return dynamics_fn

# Optional execution block for testing this module directly
if __name__ == "__main__":
    
    # Load configuration file
    config_path = "config.yaml"
    config = load_config(config_path)
    
    # Test single pendulum dynamics
    single_ddynamics = single_pendulum_dynamics(config)
    print("Single pendulum dynamics defined successfully.")
    
    # Test double pendulum dynamics
    doubel_dynamics = double_pendulum_dynamics(config)
    print("Double pendulum dynamics defined successfully.")