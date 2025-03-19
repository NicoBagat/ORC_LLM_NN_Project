import l4casadi as l4c


def single_pendulum_dynamics():    
    ''' 
        Symbolic dynamics of a single pendulum using l4casadi
        ___________________
        DYNAMICS VARAIBLES
        ___________________
        x = state
        u = control
        ___________________
    '''
    # Symbolic definition of state and control variables
    x = l4c.SX.sym("x", 2)
    u = l4c.SX.sym("u", 1)
    
    theta, omega = x[0], x[1]
    torque = u[0]
    g, L, m = 9.81, 1.0, 1.0
    
    theta_dot = omega
    omega_dot = (-m * g * L * l4c.sin(theta) + torque) / ( m * L**2)
    
    dxdt = l4c.vertcat(theta_dot, omega_dot)
    
    dynamics_fn = l4c.Function("single_pendulum_dyunamics", [x, u], [dxdt])

    return dynamics_fn

def double_pendulum_dynamics():
    """
    Define the dynamics for a double pendulum.
    Returns:
        Callable: l4casadi-compatible dynamics function.
    """
    x = l4c.SX.sym("x", 4)
    u = l4c.SX.sym("u", 1)

    # Dynamics definition omitted for brevity; similar symbolic formulation

    dynamics_fn = l4c.Function("double_pendulum_dynamics", [x, u], [dxdt])
    return dynamics_fn