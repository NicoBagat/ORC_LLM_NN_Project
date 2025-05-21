import l4casadi as l4c

def define_cost_function(config):
    """ 
    Define the stage and terminal cost functions for the OCP using l4casadi based on the configuration

    Args:
        config(dict): Configuration dictionary containing cost weights and parameters
        
    Returns:
        Tuple: A tuple containing stage_cost and terminal_cost functions compatible with l4casadi

    """
    # Extract cost weights from config file
    state_weights = config["cost_weights"]["state"]
    control_weight = config["cost_weights"]["control"]

    # Define symbolic variables
    x = l4c.SX.sym("x", config["state_dim"]) # State variables
    u = l4c.SX.sym("u", config["control_dim"]) # Control variables
    
    # Define cost functions
    stage_cost_expr = l4c.dot(state_weights, x**2) + control_weight * l4c.dot(u, u)
    terminal_cost_expr = l4c.dot(state_weights, x**2)
    
    # Convert expressions to callable functions
    stage_cost = l4c.Function("stage_cost", [x, u], [stage_cost_expr])
    terminal_cost = l4c.Function("terminal_cost", [x], [terminal_cost_expr])

    return stage_cost, terminal_cost