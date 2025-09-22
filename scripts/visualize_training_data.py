import os
import numpy as np
import matplotlib.pyplot as plt
from src.utils import load_data
import yaml

def visualize_training_data(config):
    data_path = config["paths"]["training_data"]
    data = load_data(data_path)
    x_init = np.array([d[0] for d in data])
    J_opt = np.array([d[1] for d in data])

    # Ensure 'plots' directory exists
    plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, "training_data_scatter.png")

    if x_init.shape[1] == 2:
        theta = x_init[:, 0]
        theta_dot = x_init[:, 1]
        plt.figure(figsize=(8, 6))
        sc = plt.scatter(theta, theta_dot, c=J_opt, cmap='viridis', s=80)
        plt.colorbar(sc, label="Optimal Cost J(x_init)")
        plt.xlabel("Initial Angle (theta)")
        plt.ylabel("Initial Angular Velocity (theta_dot)")
        plt.title("Training Data: Initial State vs. Optimal Cost")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        print(f"Training data plot saved to {plot_path}")
    else:
        print("Visualization for state dimension > 2 not implemented.")