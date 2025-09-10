import numpy as np

from init_net.init_starting_point import init_start_point
from init_net.process_net_data import parse_ieee_mat, System, Branch
from init_net.process_measurements import load_legacy_measurements
from init_net.power_flow_polar import H_AC
from optimizers.GN_se import GN_se

file = '../nets/ieee118_186.mat'

data = parse_ieee_mat(file)
system_data = data['data']['system']
sys = System(system_data)

legacy_data = data['data']['legacy']
z, v, meas_indexes = load_legacy_measurements(legacy_data, exact=False, sample=False,
                                              measurement_type=['flow', 'injection', 'voltage'])
branch = Branch(sys.branch)

h_ac = H_AC(sys, branch, meas_indexes)

x0 = np.r_[init_start_point(sys, data, how='flat')]
tol = 1e-10
max_iter = 500
T_est, V_est, eps, iter, converged = GN_se(z, v, sys.slk_bus, h_ac, sys.nb, tol, max_iter, x0=x0)

pmu = data['data'].get('pmu')
if pmu:
    V_true, T_true = pmu['voltage'][:, -2:].T
    sv_true = np.r_[T_true, V_true]
    sv_est = np.r_[T_est, V_est]
    dx = sv_est - sv_true
    print(f'Converged: {converged}')
    print(f'RMSE: {np.sqrt(np.sum(dx ** 2) / (2 * sys.nb)):.4e}')
    print(f'eps: {eps:.4e}')
    print()
