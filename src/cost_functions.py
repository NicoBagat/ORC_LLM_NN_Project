import numpy as np

def calculate_cost(state, control, weights):
    """ Calculate the stage cost"""
    state_cost = np.dot(weights["state"], np.square(state))
    control_cost = weights["control"] * np.sum(np.square(control))

    return state_cost + control_cost