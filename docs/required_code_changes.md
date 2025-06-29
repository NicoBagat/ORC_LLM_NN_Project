# Required Code Changes for PROJECT A

This document tracks missing features, bugs, and improvements identified during the development and review of the project.

---

## src/ocp_solver.py

### What Doesn't Work / Gaps:
1. **Control Bounds**:
    - Currently limited to fixed ranges; should adapt based on system configuration or OCP definition.
2. **State Bounds**:
    - Implementation lacks clear testing for state limits (e.g., pendulum angle limits).
3. **Solver Error Handling**:
    - No mechanism for handling solver failures or infeasible problems.
4. **Result Extraction**:
    - Results (`trajectory`, `controls`, `cost`) are returned but lack validation to ensure physical consistency (e.g., respecting dynamics).
5. **Testing**:
    - Only tested on single pendulum; double pendulum functionality is untested.

### Improvements Needed:
- Add robust error handling for solver failures.
- Validate extracted trajectories and costs for correctness.
- Extend functionality to properly handle and test double pendulum dynamics.

---

## src/dynamics.py

### What Doesn't Work / Gaps:
1. **Validation**
   - Dynamics functions are not tested against numerical or analytical benchmarks.
   - No verification that the implemented equations produce expected physical behavior.

2. **Double Pendulum Dynamics**
   - Complex interactions are defined but lack practical validation or testing in OCPs.
   - No example simulations to demonstrate correctness.

**Improvements Needed:**
- Test dynamics functions independently with numerical integration to validate correctness.
- Add examples or benchmarks for single and double pendulum simulations.

---

## src/neural_network.py

### What Doesn't Work / Gaps:
1. **Validation**
   - No evaluation of the network's performance on test data.
   - No metrics reported for prediction accuracy.

2. **Generalization**
   - Requires testing to ensure the network generalizes well to unseen states.
   - No validation against ground-truth $J(x_0)$ for states outside the training set.

3. **Scalability**
   - Network structure may need adaptation for larger state dimensions (e.g., double pendulum).
   - No experiments or code to handle increased input size.

### Improvements Needed:
- Add test dataset evaluation to measure generalization.
- Validate neural network predictions against ground-truth $J(x_0)$ for unseen states.
- Consider flexible network architectures for different system sizes.

---

## src/utils.py

### What Doesn't Work / Gaps:
1. **Error Handling**
   - No checks for missing or malformed configurations.
   - Functions like `load_config` and `load_data` do not validate input or handle exceptions robustly.

2. **Logging**
   - Lacks detailed logs for data saving/loading and normalization.
   - No traceability for key operations, making debugging harder.

### Improvements Needed:
- Add input validation for functions like `load_config` and `load_data`.
- Include logging to trace operations and debug potential issues.

---

## scripts/generate_training_data.py

### What Doesn't Work / Gaps:
1. **Initial State Sampling**
   - Currently relies only on random sampling; lacks grid-based sampling for better state space coverage.

2. **Diversity**
   - Needs to ensure diversity in training data, especially for the double pendulum.

### Improvements Needed:
- Add grid-based sampling of initial states.
- Include validation of generated data (e.g., ensure costs correspond to optimal trajectories).

---

## scripts/train_nn.py

### What Doesn't Work / Gaps:
1. **Early Stopping**
   - Lacks an early stopping mechanism to prevent overfitting.

2. **Validation**
   - Does not evaluate the network on a validation/test set during training.

3. **Logging**
   - Missing detailed logs for training progress (e.g., loss evolution).

### Improvements Needed:
- Add a validation phase during training.
- Implement early stopping based on validation performance.
- Include detailed training logs for debugging and analysis.

---

## scripts/run_mpc_with_nn.py

### What Doesn't Work / Gaps:
1. **Comparison**
   - Lacks direct comparison with a standard OCP approach (long horizon).

2. **Logging and Output**
   - Results (e.g., trajectories, controls, costs) are not clearly logged or visualized.

### Improvements Needed:
- Add functionality to compare MPC results with long-horizon OCP results.
- Include detailed result logs and trajectory visualizations.

---

## main.py

### What Doesn't Work / Gaps:
- Missing logging for intermediate steps and final results.
- Results are not stored or visualized for analysis.

### Improvements Needed:
- Add logging for each step to track progress.
- Save and visualize final results (e.g., trajectories, costs).

---

## config.yaml

### What Doesn't Work / Gaps:
1. **Incomplete Parameterization**
   - Missing parameters for grid-based sampling, logging levels, and validation data splits.

2. **Scalability**
   - Needs flexibility for larger state/control spaces (e.g., double pendulum).

### Improvements Needed:
- Add grid-based sampling parameters.
- Include validation split and logging configurations.

---

Add further sections below as new issues or improvements are identified.