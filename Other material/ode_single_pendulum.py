import numpy as np

# Constants
g = 9.81  # gravity (m/s^2)
m = 1.0   # mass of the pendulum (kg)
L = 1.0   # length of the pendulum (m)

def pendulum_dynamics(state, u=0.0):
    """
    Compute the dynamics of the single pendulum.
    
    Parameters:
    - state: [theta, omega], where:
      - theta is the angle (rad)
      - omega is the angular velocity (rad/s)
    - u: control input, e.g., torque (not used here but can be included if needed)

    Returns:
    - dynamics: [dtheta/dt, domega/dt]
    """
    theta, omega = state
    domega = -(g / L) * np.sin(theta) + u / (m * L**2)  # Including u for extensibility
    return np.array([omega, domega])

def runge_kutta_step(state, dt, u=0.0):
    """
    Perform a single step of Runge-Kutta integration for the pendulum.
    
    Parameters:
    - state: current state [theta, omega]
    - dt: time step
    - u: control input (optional)

    Returns:
    - new_state: updated state after time step
    """
    k1 = pendulum_dynamics(state, u)
    k2 = pendulum_dynamics(state + 0.5 * dt * k1, u)
    k3 = pendulum_dynamics(state + 0.5 * dt * k2, u)
    k4 = pendulum_dynamics(state + dt * k3, u)
    
    new_state = state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return new_state
