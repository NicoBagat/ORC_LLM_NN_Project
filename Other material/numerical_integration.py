# Constains a numerical integration function (Euler integration) to simulate pendulum dynamics

import torch
from Project.A3.ode_single_pendulum import pendulum_dynamics
def euler_integration(x, u, dt=0.01):
    dx = pendulum_dynamics(x, u)
    next_x = x +dx * dt
    return next_x

def integrate_dynamics(x, u, method='euler'):
    if method == 'euler':
        return euler_integration(x, u)
    else:
        raise ValueError("Unknown integration method")