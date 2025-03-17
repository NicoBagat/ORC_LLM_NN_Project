from l4casadi 
import SX, Function

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
    
    # Extract cost weights from config
    x = SX.sym("x", config["state_dim"]) # State variables
    u = SX.sym("u", config["control_dim"]) # control variables
    
    # Stage cost: sum of quadratic state and control penalties
    stage_cost_expr = 
    
    # Terminal cost: quadratic state penalty
    terminal_cost_expr = 
    
    # Define CasADi functions
    stage_cost = 
    terminal_cost = 
    return stage_cost, terminal_cost