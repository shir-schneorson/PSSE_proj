import numpy as np
from scipy.io import loadmat

from compute_power_flow import compute_power_flow
from gauss_newton_state_estimation_linesearch import gauss_newton_state_estimation_linesearch
from gauss_newton_state_estimation_real import gauss_newton_state_estimation_real
from generate_Ybus import generate_Ybus

# Load IEEE 118-bus data
data = loadmat('ieee118_186.mat')  # Ensure this file contains 'data.system.line'

data['data'] = {n: data['data'][n][0, 0] for n in data['data'].dtype.names}
data['data']['system'] = {n: data['data']['system'][n][0, 0] for n in data['data']['system'].dtype.names}

# Extract line and bus data
line_data = data['data']['system']['line'] # Adjust indexing if necessary
n = data['data']['system']['bus'].shape[0]  # Number of buses

# Generate Ybus
Ybus = generate_Ybus(n, line_data)

# Define slack bus
slack_bus = 0  # Assuming 1-based indexing in MATLAB, adjust to 0-based in Python

# Generate true power injections
theta_true = np.deg2rad(10 * (np.random.rand(n) - 0.5))  # Small angle deviations
theta_true -= theta_true[slack_bus]  # Set slack bus angle to 0
V_true = 1 + 0.05 * (np.random.rand(n) - 0.5)

# Compute true power injections
P_meas, Q_meas = compute_power_flow(theta_true, V_true, np.real(Ybus), np.imag(Ybus))

# Add measurement noise
P_meas += 0.01 * np.random.randn(*P_meas.shape)
Q_meas += 0.01 * np.random.randn(*Q_meas.shape)

# Set convergence tolerance and max iterations
tol = 1e-6
max_iter = 100

# Perform Gauss-Newton state estimation
theta_est, V_est = gauss_newton_state_estimation_real(P_meas, Q_meas, Ybus, slack_bus, tol, max_iter)
theta_est2, V_est2 = gauss_newton_state_estimation_linesearch(P_meas, Q_meas, Ybus, slack_bus, tol, max_iter)

# Display results
print('Estimated Voltage Angles (radians):')
print(theta_est)
print('Estimated Voltage Magnitudes (p.u.):')
print(V_est)
