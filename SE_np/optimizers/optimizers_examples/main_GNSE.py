import numpy as np

from SE_np.utils import init_start_point
from SE_np.net_preprocess.process_net_data import parse_ieee_mat, System, Branch
from SE_np.net_preprocess.process_measurements import load_legacy_measurements
from SE_np.PF_equations.PF_polar import H_AC
from SE_np.optimizers import GN_se

file = '../../../nets/ieee118_186.mat'

data = parse_ieee_mat(file)
system_data = data['data']['system']
sys = System(system_data)

legacy_data = data['data']['legacy']
z, v, meas_indexes = load_legacy_measurements(legacy_data, exact=False, sample=False,
                                              measurement_type=['flow', 'injection', 'voltage', 'current'])
branch = Branch(sys.branch)

h_ac = H_AC(sys, branch, meas_indexes)

x0 = np.r_[init_start_point(sys, data, how='flat')]
tol = 1e-10
max_iter = 500
x, x_list, converged, k = GN_se(x0, z, v, sys.slk_bus, h_ac, sys.nb,tol=tol, max_iter=max_iter)
T_est, V_est = x[:sys.nb], x[sys.nb:]

pmu = data['data'].get('pmu')
if pmu:
    V_true, T_true = pmu['voltage'][:, -2:].T
    sv_true = np.r_[T_true, V_true]
    sv_est = np.r_[T_est, V_est]
    dx = sv_est - sv_true
    print(f'Converged: {converged}')
    print(f'RMSE: {np.sqrt(np.sum(dx ** 2) / (2 * sys.nb)):.4e}')
    # print(f'eps: {eps:.4e}')
    print()
