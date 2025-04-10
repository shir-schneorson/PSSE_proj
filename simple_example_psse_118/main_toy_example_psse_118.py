import numpy as np
from scipy.io import loadmat

from gauss_newton_state_estimation_linesearch import gauss_newton_state_estimation_linesearch
from simple_example_psse_118.gauss_newton_state_estimation_real import gauss_newton_state_estimation_real
from generate_Ybus import generate_Ybus
from simple_example_psse_118.compute_power_flow import compute_power_flow
np.random.seed(42)
# Load IEEE 118-bus data
data = loadmat('simple_example_psse_118/ieee118_186.mat')  # Ensure this file contains 'data.system.line'

data['data'] = {n: data['data'][n][0, 0] for n in data['data'].dtype.names}
data['data']['system'] = {n: data['data']['system'][n][0, 0] for n in data['data']['system'].dtype.names}
data['data']['pmu'] = {n: data['data']['pmu'][n][0, 0] for n in data['data']['pmu'].dtype.names}
data['data']['legacy'] = {n: data['data']['legacy'][n][0, 0] for n in data['data']['legacy'].dtype.names}

# Extract line and bus data`
line_data = data['data']['system']['line'] # Adjust indexing if necessary
bus_data = data['data']['system']['bus']
n = bus_data.shape[0]  # Number of buses
nl = line_data.shape[0]

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
# P_meas += 0.01 * np.random.randn(*P_meas.shape)
# Q_meas += 0.01 * np.random.randn(*Q_meas.shape)

# Set convergence tolerance and max iterations
tol = 1e-8
max_iter = 100

# Perform Gauss-Newton state estimation
theta_est, V_est, _ = gauss_newton_state_estimation_real(P_meas, Q_meas, Ybus, slack_bus, tol, max_iter)
theta_est2, V_est2, _ = gauss_newton_state_estimation_linesearch(P_meas, Q_meas, Ybus, slack_bus, tol, max_iter)

# Display results
print('**Errors**')
print(f'Voltage error GN real: {np.linalg.norm(V_est - V_true):.6f}')
print(f'Voltage error GN linesearch: {np.linalg.norm(V_est2 - V_true):.6f}')
print(f'Theta (radians) error GN real: {np.linalg.norm(theta_est - theta_true):.6f}')
print(f'Theta (radians) error GN linesearch: {np.linalg.norm(theta_est2 - theta_true):.6f}')
print()

print('Estimated Voltage Angles (radians):')
print(f'GN real: {theta_est}')
print(f'GN linesearch: {theta_est2}')
print()
print('Estimated Voltage Magnitudes (p.u.):')
print(f'GN real: {V_est}')
print(f'GN linesearch: {V_est2}')


