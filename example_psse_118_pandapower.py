import numpy as np

import pandapower as pp
from pandapower.networks.power_system_test_cases import case118


# Load IEEE 118-bus data
ieee118_net = case118()

# Compute power flow in net
pp.runpp(ieee118_net)

# add P, Q measurements
for i, bus_pf in ieee118_net['res_bus'].iterrows():
    p_mw, q_mvar = bus_pf['p_mw'], bus_pf['q_mvar']
    p_mw += 0.01 * np.random.randn()
    q_mvar += 0.01 * np.random.randn()
    pp.create_measurement(ieee118_net, 'p', 'bus', p_mw, 0.01, i)
    pp.create_measurement(ieee118_net, 'q', 'bus', q_mvar, 0.01, i)


for i, line_pf in ieee118_net['res_line'].iterrows():
    p_from_mw, q_from_mvar = line_pf['p_from_mw'], line_pf['q_from_mvar']
    p_to_mw, q_to_mvar = line_pf['p_to_mw'], line_pf['q_to_mvar']

    p_from_mw += 0.01 * np.random.randn()
    q_from_mvar += 0.01 * np.random.randn()

    p_to_mw += 0.01 * np.random.randn()
    q_to_mvar += 0.01 * np.random.randn()

    pp.create_measurement(ieee118_net, 'p', 'line', p_from_mw, 0.01, i, side='from')
    pp.create_measurement(ieee118_net, 'p', 'line', p_to_mw, 0.01, i, side='to')
    pp.create_measurement(ieee118_net, 'q', 'line', q_from_mvar, 0.01, i, side='from')
    pp.create_measurement(ieee118_net, 'q', 'line', q_to_mvar, 0.01, i, side='to')


# Perform WLS state estimation
pp.estimation.estimate(ieee118_net)

# Display results
theta_est, V_est = ieee118_net['res_bus_est'].loc[:, ['va_degree', 'vm_pu']].values.T
theta_true, V_true = ieee118_net['res_bus'].loc[:, ['va_degree', 'vm_pu']].values.T

print('**Errors**')
print(f'Voltage error WLS pandapower: {np.linalg.norm(V_est - V_true):.6f}')
print(f'Theta (radians) error WLS pandapower: {np.linalg.norm(theta_est - theta_true):.6f}')
print()

print('Estimated Voltage Angles (radians):')
print(np.deg2rad(theta_est))
print('Estimated Voltage Magnitudes (p.u.):')
print(V_est)