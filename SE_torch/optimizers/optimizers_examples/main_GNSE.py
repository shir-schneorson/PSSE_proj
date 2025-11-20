import torch

from SE_torch.optimizers.FO_se import SGD_se
from SE_torch.utils import init_start_point, RMSE
from SE_torch.net_preprocess.process_net_data import parse_ieee_mat, System, Branch
from SE_torch.net_preprocess.process_measurements import load_legacy_measurements
from SE_torch.PF_equations.PF_polar import H_AC
from SE_torch.optimizers.GN_se import GN_se

file = '../../../nets/ieee118_186.mat'

data = parse_ieee_mat(file)
system_data = data['data']['system']
sys = System(system_data)

legacy_data = data['data']['legacy']
z, v, meas_indexes = load_legacy_measurements(legacy_data, exact=False, sample=False,
                                              measurement_type=['flow', 'injection', 'voltage'])
branch = Branch(sys.branch)

h_ac = H_AC(sys, branch, meas_indexes)

x0 = torch.cat(init_start_point(sys, data, how='flat'))
tol = 1e-10
max_iter = 10000000
optimizer = GN_se(tol=tol, max_iter=max_iter)
x_gn, T_gn, V_gn, gn_converged, k_gn = optimizer(x0, z, v, sys.slk_bus, h_ac, sys.nb)


pmu = data['data'].get('pmu')
V_true, T_true = torch.tensor(pmu['voltage'][:, -2:]).T

print(f'Converged: {gn_converged}')
print(f'RMSE: {RMSE(T_true, V_true, T_gn, V_gn):.4e}')
print(f'Steps: {k_gn}')
print()
