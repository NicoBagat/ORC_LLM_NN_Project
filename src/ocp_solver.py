import l4casadi as l4c
import casadi as cs
from src.dynamics import single_pendulum_dynamics, double_pendulum_dynamics

def define_ocp(config, dynamics, stage_cost, terminal_cost):
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
    horizon = config["ocp"]["horizon"]
    dt = config["ocp"]["dt"]
    cost_weights = config["ocp"]["cost_weights"]
    
    # Define symbolic variables
    x = cs.MX.sym("x", state_dim)   # State variable
    u = cs.MX.sym("u", control_dim) # Control variable
    
    # Dynamics definition (based on active dynamic in config file)
    dynamics_fn = (
        single_pendulum_dynamics(config)
        if config["ocp"]["dynamics"] == "single_pendulum"
        else double_pendulum_dynamics(config)
    )
    
    f = lambda x, u: dynamics_fn(x, u).reshape(-1)
    
    # Define the OCP structure
    ocp = l4c.OptimalControlProblem()
    ocp.add_state(x)
    ocp.add_control(u)
    ocp.set_dynamics(f, dt=dt)
    
    # Set objective function
    Q = cs.diag(cs.MX(cost_weights["state"]))
    R = cs.MX(cost_weights["control"])
        
    # Set stage cost and terminal cost
    
    # Stage cost
    def stage_cost(x, u):
        return cs.mtimes([x.T, Q, x]) + cs.mtimes([u.T, R, u])
    
    ocp.set_Stage_cost(stage_cost)
    
    # Terminal cost
    def terminal_cost(x):
        return cs.tmtime([x.T, Q, x])
    
    ocp.set_terminal_cost(terminal_cost)
    
    # Add control bounds (if specified in config)
    if "control_bounds" in config["ocp"]:
        u_min, u_max = config["ocp"]["control_bounds"]
        ocp.add_control_bound(u_min, u_max)
        
    # Add state bounds (if specified in config)
    if "state_bounds" in config["ocp"]:
        x_min, x_max = config["ocp"]["state_bounds"]
        ocp.add_state_bound(x_min, x_max)
    
    # Set solver options
    solver_options = {
        "ipopt.print_level":0,
        "print_time": False        
    }
    ocp.set_solver_options(solver_options)
    
    return ocp, x, u, dynamics_fn

def solve_ocp(initial_state, config):
    '''
    Solve the OCP for a given initial state.
    
        Args:
            initial_state (numpy.ndarray): Initial state vector
            config (dict): Cofiguration dictionary loaded from 'config.yaml'
            
        Returns:
            dict: A dictionary containing the optimal trajecotry, control sequence and total cost.
    '''
    # Define the OCP
    ocp, x, u, dynamics_fn = define_ocp(config)
    
    # Set the initial state
    ocp.set_initial_state(initial_state)
    
    # Solve the OCP
    solver = ocp.get_solver()
    solution = solver.solve()
    
    # Extract the results
    trajectory = solution.get_state_trajectory()
    controls =  solution.get_control_trajectory()
    cost = solution.get_total_cost()
    
    return {
            "trajectory": trajectory,
            "controls": controls,
            "cost": cost
            }