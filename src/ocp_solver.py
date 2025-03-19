import l4casadi as l4c

def define_ocp(config, dynamics, stage_cost, terminal_cost):
    ''' 
    Define the OCP using l4casadi 
        
        Args:
            config (dict): Configuration dictionary
            dynamics (function): Dynamics function
            stage_cost (function): Stage cost function
            terminal_cost (function): Terminal cost function
            
        Return:
            l4c.OptimalControlProblem: Configured OCP instance
    '''
    # Initialize the OCP
    ocp = l4c.OptimalControlProblem(
        n_states = config["state_dim"],
        n_controls=config["control_dim"],
        horizon=config["horizon"],
        dt=config["dt"],
    )
    
    # Dynamics
    ocp.set_dynamics(dynamics)
    
    # Set stage cost and terminal cost
    ocp.set_cost(stage_cost=stage_cost, terminal_cost=terminal_cost)
    
    return ocp

def solve_ocp(ocp, x0):
    '''
    Solve the OCP for a given initial state.
    
        Args:
            ocp (l4c.OptimalControlProblem): Configured OCP instance
            x0 (lsit or ndarray): Initial state
            
        Returns:
            tuple: A tuple containing trajectory, controls and cost.
    '''
    trajectory, controls, cost = ocp.solve(x0)
    return trajectory, controls, cost