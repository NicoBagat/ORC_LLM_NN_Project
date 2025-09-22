import numpy as np
from src.define_ocp import define_ocp
from src.utils import save_data, load_config

def generate_training_data(config):
    """
    Generate training data by solving OCPs:
    - horizon N
    - no terminal cost
    - no inequality constraints
    """
    
    N = config["ocp"]["horizon" ] # Long horizon for value function learning
    test_states = config["training"]["initial_states"] # List of initial states (random or grid)
    training_data = []
    
    for x_init in test_states:
        # Define OCP for this initial state
        ocp, x, u, dynamics_fn = define_ocp(config)
        dt = ocp["dt"]
        state_dim = x.size1()
        control_dim = u.size1()
        
        # Devision variables
        X = ocp["cs"].MX.sym('X', state_dim, N+1)  # States over the horizon
        U = ocp["cs"].MX.sym('U', control_dim, N)  # Controls over the horizon
        
        cost = 0 
        constraints = []
        constraints.append(X[:,0] - x_init)
        
        for k in range(N):
            x_k = X[:, k]
            u_k = U[:, k]
            x_next = X[:, k+1]
            x_next_pred = x_k + dt * ocp["dynamics_fn"](x_k, u_k)   
            constraints.append(x_next - x_next_pred)
            cost += ocp["stage_cost"](x_k, u_k)

        g = ocp["cs"].vertcat(*constraints)
        vars = ocp["cs"].vertcat(ocp["cs"].reshape(X, -1, 1), ocp["cs"].reshape(U, -1, 1))
        nlp = {'x': vars, 'f': cost, 'g': g}
        solver = ocp["cs"].nlpsol('solver', 'ipopt', nlp)
        
        # Initial guess
        x0 = [float(v) for v in x_init] * (N+1)
        u0 = [0.0] * (control_dim * N)
        vars_init = x0 + u0
        sol = solver(x0=vars_init, lbg=0, ubg=0)
        J_opt = float(sol['f'].full().item())
        
        training_data.append((np.array(x_init), J_opt))
        
    # Save training 
    save_data(np.array(training_data, dtype=object), config["paths"]["training_data"])
    print(f"Generated {len(training_data)} data points.")
    return training_data
