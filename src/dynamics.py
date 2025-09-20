import casadi as cs
import numpy as np

def single_pendulum_dynamics(config):
    """
    Return a CasADi function f(x, u) for SINGLE PENDULUM DYNAMICS:
    x = [theta, theta_dot]
    u = [torque]
    """
    
    g = 9.81 # gravity
    l = config.get("penduylum_length", 1.0) # length of the pendulum
    m = config.get("pendulum_mass", 1.0)   # mass of the pendulum
    b = config.get("pendulum_damping", 0.1) # damping coefficient
    
    x =cs.MX.sym("x", 2)
    u =cs.MX.sym("u", 1)
    
    theta = x[0]
    theta_dot = x[1]
    torque = u[0]
    
    theta_ddot = (torque - b * theta_dot - m * g * l * cs.sin(theta)) / (m * l ** 2)
    x_dot = cs.vertcat(theta_dot, theta_ddot)
    
    return cs.Function("f", [x, u], [x_dot])

def double_pendulum_dynamics(config):
    """ Returns a CasADi function f(x, u) for double pendulum dynamics.
    
    x = [theta1, theta2, theta1_dot, theta2_dot]
    u = [torque]
    """
    
    # Example parameters (customize as needed)
    g = 9.81  # gravity
    l1 = config.get("pendulum1_length", 1.0)  # length of the first pendulum
    l2 = config.get("pendulum2_length", 1.0)  # length of the second pendulum
    m1 = config.get("pendulum1_mass", 1.0)    # mass of the first pendulum
    m2 = config.get("pendulum2_mass", 1.0)    # mass of the second pendulum
    b1 = config.get("pendulum1_damping", 0.1) # damping coefficient for first pendulum
    b2 = config.get("pendulum2_damping", 0.1) # damping coefficient for second pendulum
    
    x = cs.MX.sym("x", 4)
    u = cs.MX.sym("u", 1)
    
    theta1 = x[0]
    theta2 = x[1]
    theta1_dot = x[2]
    theta2_dot = x[3]
    torque = u[0]
    
    # Equations for double pendulum (simpliefied, planar, actuated at first joint)
    delta = theta2 - theta1
    den1 = (m1 +m2) *l1 -m2 *l1 * cs.cos(delta) * cs.cos(delta) * cs.cos(delta)
    den2 = (l2 / l1) * den1
    
    theta1_ddot = (
        m2 * l1 * theta1_dot ** 2 * cs.sin(delta) * cs.cos(delta) 
        + m2 * g * cs.sin(theta2) * cs.cos(delta)
        + m2 * l2 * theta2_dot ** 2 * cs.sin(delta)
        -(m1 + m2) * g * cs.sin(theta1)
        + torque # torque applied only at first joint
        - b1 * theta1_dot
    ) / den1
    
    theta2_ddot = (
        -m2 * l2 * theta2_dot ** 2 * cs.sin(delta) * cs.cos(delta) 
        + (m1 + m2) * g * cs.sin(theta2) * cs.cos(delta)
        - (m1 + m2) * l2 * theta2_dot ** 2 * cs.sin(delta)
        - (m1 + m2) * g * cs.sin(theta1)
        # torque applied only at first joint
        - b2 * theta2_dot
    ) / den2

    x_dot = cs.vertcat(theta1_dot, theta2_dot, theta1_ddot, theta2_ddot)
    
    return cs.Function("f", [x, u], [x_dot])