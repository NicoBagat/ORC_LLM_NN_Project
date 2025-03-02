AIM:Use a large language model powered neural network to estimate a value function for an optimal control problem

PROJECT DEVELOPEMENT STEPS:
    1. PROBLEM FORMULATION
        '1. Define the OCP (Optimal Control Problem) for the single pendulum
        '2. Specify state dynamics, cost function and constraints
        '3. Decide the horizon length for the initial OCP and the reduced horizon for testing the terminal cost

    2. DATA GENERATION
        '1. Solve the multiple OCPs starting from differential initial states x0 (random or grid-based sampling).
        '2. Store each x0 and the corresponding optimal cost J(x0) in a dataset


    3. NEURAL NETWORK DESIGN AND TRAINING
        '1. Design a neural network in PyTorch to predict J(x0) given x0
        '2. Train the network using the dataset generated in Step 2
        '3. Evaluate its performance using validation/test sets

    4. INTEGRATION INTO MPC
        '1. Incorporate the trained neural network as a terminal cost into the OCP formulation with a shorter horizon
        '2. Solve the new OCPs using the modified cost function

    5. VALIDATION
        '1. Compare the performance of the shortened-horizon MPC with and without the neural network terminal cost
        '2. Analyze metrics like computation time, control performance and convergence

    6. EXTENSION TO DOUBLE PENDULUM
        '1. Adapt the OCP formulation, data generation and training process for the double pendulum

_________________________________________________________________________________________________________________________________________________________________
ocp_solver.py

    - Define the OCP formulation for the single and double pendulum
    - Solve the OCPSs using l4casadi
_________________________________________________________________________________________________________________________________________________________________
data_generation.py

    - Generate the initial states x0
    - Solve the OCPs
    - Store the dataset (x0,J(x0))
_________________________________________________________________________________________________________________________________________________________________
neural_network.py

    - Define and train the PyTorch neural network model for predicting J(x0)
    - Save and load the trained model
_________________________________________________________________________________________________________________________________________________________________
evaluate_network.py

    - Test the performance of the neural network on unseen data
    - Generate visualizations like loss curves or predictions vs true values
_________________________________________________________________________________________________________________________________________________________________
mcp_with_terminal_cost.py

    - Implement the MPC formualtion with the neural network as the terminal mcp_with_terminal_cost
    - Solve the new OCPs with a shorter horizon
_________________________________________________________________________________________________________________________________________________________________
validation.py

    - Compare results between the full-horizon MPC and the shortened-horizon MPC with the terminal cost
_________________________________________________________________________________________________________________________________________________________________
utils.py

    - Utility functions for shared tasks like saving/loading models, handling datasets and plotting
_________________________________________________________________________________________________________________________________________________________________
config.py

    - Centralized configuration file for specifying parameters like pendulum dynamics, neural network hyperparameters and solver settings
_________________________________________________________________________________________________________________________________________________________________

