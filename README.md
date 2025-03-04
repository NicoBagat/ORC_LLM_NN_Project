#   Value function approximation for MPC
This repository implements a method to approximate a Value function using a neural network, 
which is used as a terminal cost in Model Predictive Control (MPC).
The method is tested on single and double pendulum syustems.

##   Workflow
1. Solve multiple OCPs to generate training data
2. Train a neural nwtowkr to predict the optimal cost-to-go
3. Use the trained network as a terminal cost in MPC
4. Empirically compare results between standard MPC nad shortened-horizon MPC with the neural network terminal cost

## Repository Structure
- `config.yaml` : central configuration file for the project
- `main.py` : mains script to run the workflow
- `data/` : directory for raw data and processed results
- `docs/` : assignment requests / project developement overview
- `models/` : constains trained neural network models 
- `src/`: source code directory with dynamics, OCP solversm, neural network implementations and utilities

## Installation
1. Clone the repository:

    `git clone <repository-url>`

    `cd <repository-name>`

2. Install dependencies:

    `pip install -r requirements.txt`

## Usage
1. Configure the project parameters in `config.yaml`
2. Run the workflow

    `python main.py`