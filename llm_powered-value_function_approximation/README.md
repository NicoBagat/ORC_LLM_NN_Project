
AIM:Use a large language model (LLM) powered neural network to estimate a cost function for an optimal control problem

* PROJECT DEVELOPEMENT STEPS:

    1. PROBLEM FORMULATION
        * Write functions for both single and double pendulum dynamics to define their equations of motion

    2. DATA GENERATION
        * Randomly or systematically generate initial states
        * Solve the OCPs to compute the optimal costs J(x0)
        * Save the initial states and costs to a dataset

    3. NEURAL NETWORK DESIGN AND TRAINING
        * Design a neural network in PyTorch to approximate J(x0)
        * Use the dataset to train the network and save the model

    4. INTEGRATION INTO MPC
        * Modify OCP fdrmulation to use the neural network as the terminal cost
        * Shorten the horizon and verify that the terminal cost compensates for the reduced horizon

    5. EVALUATION
        * Test the modifie OCP with several initial states
        * Save and analyze the results to verify for performance improvements
_____________________________________________________________________________________________________________________________________________________________________
* REPOSITORY STRUCTURE

    llm_powered-value_function_approximation/
    ├── main.py
    ├── config.yaml
    ├── requirements.txt
    ├── docs # documents and project request
    ├── data/
    │   ├── raw/  # Input data or initial states for training
    │   ├── processed/  # Outputs like datasets, results, etc.
    ├── models/
    │   └── neural_network.pt  # Trained neural network weights
    ├── src/
    │   ├── dynamics.py
    │   ├── ocp_solver.py
    │   ├── neural_network.py
    │   ├── data_processing.py
    │   └── training.py
    └── tests/
        └── test_ocp_solver.py
    _____________________________________________________________________________________________________________________________________________________________________

* MAIN FILES
    _____________________________________________________________________________________________________________________________________________________________________
    * main.py
     ____________________________________________________________________________________________________________________________________________________________________
    * config.yaml
    _____________________________________________________________________________________________________________________________________________________________________
    * rrequirements.txt
     ____________________________________________________________________________________________________________________________________________________________________
    * dynamics.py
    _____________________________________________________________________________________________________________________________________________________________________
    * ocp_solve.py
     ____________________________________________________________________________________________________________________________________________________________________
    * neural_network.py
    _____________________________________________________________________________________________________________________________________________________________________
    * data_processing.py
    _____________________________________________________________________________________________________________________________________________________________________
    * training.py
    _____________________________________________________________________________________________________________________________________________________________________
    * test_ocp_solve.py