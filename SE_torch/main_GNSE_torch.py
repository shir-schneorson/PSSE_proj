import torch
import numpy as np

from SE_torch.init_starting_point_torch import init_start_point
from SE_torch.process_net_data_torch import parse_ieee_mat, System, Branch
from SE_torch.process_measurements_torch import load_legacy_measurements
from SE_torch.power_flow_polar_torch import H_AC
from SE_torch.GN_se_torch import GN_se_torch

file = 'nets/ןן'

data = parse_ieee_mat(file)
system_data = data['data']['system']
sys = System(system_data)

legacy_data = data['data']['legacy']
z, v, meas_indexes = load_legacy_measurements(legacy_data, exact=False, sample=False,
                                              measurement_type=['flow', 'injection', 'voltage', 'current'])
branch = Branch(sys.branch)

h_ac = H_AC(sys, branch, meas_indexes)

x0 = torch.cat(init_start_point(sys, data, how='flat'))
tol = 1e-7
max_iter = 500
T_est, V_est, eps, iter, converged = GN_se_torch(z, v, sys.slk_bus, h_ac, sys.nb, tol, max_iter, x0=x0)

pmu = data['data'].get('pmu')
if pmu:
    V_true, T_true = torch.tensor(pmu['voltage'][:, -2:]).T
    sv_true = torch.cat([T_true, V_true])
    sv_est = torch.cat([T_est, V_est])
    dx = sv_est - sv_true
    print(f'Converged: {converged}')
    print(f'RMSE: {torch.sqrt(torch.sum(dx ** 2) / (2 * sys.nb)):.4e}')
    print(f'eps: {eps:.4e}')
    print()
