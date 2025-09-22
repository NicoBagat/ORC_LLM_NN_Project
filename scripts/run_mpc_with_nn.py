import casadi as cs
import l4casadi as l4c
from src.define_ocp import define_ocp
from src.utils import load_config

import numpy as np
import torch
from src.neural_network import NeuralNetwork    

def run_mpc_with_nn(config_path="config.yaml"):
    """ 
    Run MPC using a neural network as the terminal cost (symcolically via l4casadi).
    
    Args:
        config_path(str): Path to the configuration file.
    
    Returns:
        lists: Results from MPC simulations (states, controls, costs).
    """

    # Load configuration
    config = load_config(config_path)
    
    # Load trained neural network model

    nn_model = NeuralNetwork.from_config(config_path)
    nn_model.load_state_dict(torch.load(config["paths"]["model"], map_location="cpu"))
    nn_model.eval()

    # Create CasADi function from PyTorch model using l4casadi
    input_dim = config["ocp"]["state_dim"]
    state = cs.MX.sym("x", input_dim)
    l4c_model = l4c.L4CasADi(nn_model, device='cpu')
    nn_casadi = cs.Function('nn_func', [state], [l4c_model(state)])
    
    results = []
    ocp, x, u, dynamics_fn = define_ocp(config)
    M = config["mpc"]["short_horizon"] # MPC horizon (from config file)
    dt = ocp["dt"] # Time step (from config file)
    state_dim = x.size1() # State dimension
    control_dim = u.size1() # Control dimension
    
    for test_state in config["mpc"]["test_states"]:
        # Decision variables
        X = cs.MX.sym("X", state_dim, M+1) # States over the horizon
        U = cs.MX.sym("U", control_dim, M)   # Controls over the horizon\
            
        cost = 0
        constraints = []
        constraints.append(X[:, 0] - test_state) # Initial state constraint
        
        for k in range(M):
            
            #Dynamics constraint
            x_k = X[:, k] # Current state
            u_k = U[:, k] # Current control
            
            x_next = X[:, k+1] # Next state
            x_next_pred = x_k + dt * dynamics_fn(x_k, u_k) # Predicted next state using dynamics
            constraints.append(x_next - x_next_pred) # Dynamics constraint
            cost += ocp["stage_cost"](x_k, u_k) # Accumulate stage cost
        
        # Add symbolic neural network terminal cost
        x_terminal = X[:, M]
        terminal_cost = nn_casadi(cs.reshape(x_terminal, (1, -1)))[0, 0] # Reshape for single input))
        cost += terminal_cost
        
        g = cs.vertcat(*constraints) # Concatenate all constraints
        vars = cs.vertcat(cs.reshape(X, -1, 1), cs.reshape(U, -1, 1)) # Decision variables vector
        nlp = {'x': vars, 'f': cost, 'g' :g}
        solver = cs.nlpsol('solver', 'ipopt', nlp)
        
        # Initial guess and bounds
        x0 = [float(v) for v in test_state] * (M+1)
        u0 = [0.0]* (control_dim * M)
        vars_init = x0 + u0
        
        sol = solver(x0=vars_init, lbg=0, ubg=0)
        w_opt = sol['x'].full().flatten()
        X_opt = w_opt[:state_dim * (M+1)].reshape((state_dim, M+1))
        U_opt = w_opt[state_dim * (M+1):].reshape((control_dim, M))
        
        # Evaluate terminal cost numerically for reporting
        x_terminal_val = X_opt[:, -1]
        terminal_cost_val = float(nn_casadi(np.array(x_terminal_val, dtype=np.float32).reshape(1, -1))[0, 0])
        
        results.append({
            "trajectory": X_opt,
            "controls": U_opt,
            "cost": float(sol['f']),
            "terminal_cost": terminal_cost_val  
        })
        
    return results
    