import casadi as cs
import l4casadi as l4c
import numpy as np
import torch
from src.define_ocp import define_ocp
from src.utils import load_config
from src.neural_network import NeuralNetwork    

def run_mpc_with_nn(config_path="config.yaml"):
    """ 
    Run MPC using a neural network as the terminal cost (symbolically via l4casadi).
    """

    # Load configuration
    config = load_config(config_path)
    
    # Load trained neural network model
    nn_model = NeuralNetwork.from_config(config_path)
    nn_model.load_state_dict(torch.load(config["paths"]["model"], map_location="cpu"))
    nn_model.eval()

    # Script the model for l4casadi
    scripted_nn = torch.jit.script(nn_model)
    l4c_model = l4c.L4CasADi(scripted_nn, device='cpu')

    # Create CasADi function for terminal cost
    state_dim = config["ocp"]["state_dim"]
    state_sym = cs.MX.sym("x", state_dim)
    nn_casadi = cs.Function('nn_func', [state_sym], [l4c_model(state_sym)])

    # Prepare OCP
    ocp, x, u, dynamics_fn = define_ocp(config)
    M = config["mpc"]["short_horizon"]
    dt = ocp["dt"]
    control_dim = u.size1()

    results = []

    for test_state in config["mpc"]["test_states"]:
        # Decision variables
        X = cs.MX.sym("X", state_dim, M+1)
        U = cs.MX.sym("U", control_dim, M)
        
        # Constraints and cost
        constraints = []
        cost = 0
        
        # Initial state constraint
        constraints.append(X[:, 0] - cs.vertcat(*test_state))
        
        for k in range(M):
            x_k = X[:, k]
            u_k = U[:, k]
            x_next = X[:, k+1]

            # Predict next state using symbolic dynamics
            x_next_pred = x_k + dt * dynamics_fn(x_k, u_k)  # dynamics_fn must return MX
            constraints.append(x_next - x_next_pred)

            # Stage cost
            cost += ocp["stage_cost"](x_k, u_k)

        # Terminal cost from neural network
        x_terminal = X[:, M]
        cost += nn_casadi(x_terminal) # Let CasADi handle MX

        # Build NLP
        g = cs.vertcat(*constraints)
        vars = cs.vertcat(cs.reshape(X, -1, 1), cs.reshape(U, -1, 1))
        nlp = {'x': vars, 'f': cost, 'g': g}
        solver = cs.nlpsol('solver', 'ipopt', nlp)

        # Initial guess
        x0 = np.tile(np.array(test_state, dtype=float), M+1)
        u0 = np.zeros(control_dim * M)
        vars_init = np.concatenate([x0, u0])

        # Solve NLP
        sol = solver(x0=vars_init, lbg=0, ubg=0)
        w_opt = sol['x'].full().flatten()
        X_opt = w_opt[:state_dim * (M+1)].reshape((state_dim, M+1))
        U_opt = w_opt[state_dim * (M+1):].reshape((control_dim, M))

        # Evaluate terminal cost numerically
        x_terminal_val = X_opt[:, -1]
        x_terminal_dm = cs.DM(x_terminal_val)
        terminal_cost_val = float(nn_casadi(x_terminal_dm))

        results.append({
            "trajectory": X_opt,
            "controls": U_opt,
            "cost": float(sol['f']),
            "terminal_cost": terminal_cost_val
        })

    return results
