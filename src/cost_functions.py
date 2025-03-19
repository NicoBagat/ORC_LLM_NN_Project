from l4casadi import SX
from l4casadi import Function

def define_cost_function(config):
    """ 
    Define the stage and terminal cost functions for the OCP using l4casadi

    Args:
        config(dict): Configuration dictionary with cost parameters
        
    Returns:
        stage_cost (Function): CasADi Function for the stage cost
        terminal_cost (Function): CasADi Function for the terminal cost
    """
    
    # Define symbolic variables
    x = SX.sym("x", config["state_dim"]) # State variables
    u = SX.sym("u", config["control_dim"]) # control variables
    
    # Extract cost weights from config
    state_weights = config["cost_weights"]["state"]
    control_weight = config["cost_weights"]["control"]
    
    # Stage cost: sum of quadratic state and control penalties
    stage_cost_expr = (
        sum(state_weights * x**2 + control_weight * sum(u**2))
    )
    
    # Terminal cost: quadratic state penalty
    terminal_cost_expr = sum(state_weights * x**2)
    
    # Define CasADi functions
    stage_cost = Function("stage_cost", [x, u],  [stage_cost_expr])
    terminal_cost = Function("terminal_cost", [x], [terminal_cost_expr])
    
    return stage_cost, terminal_cost