import numpy as np

from power_flow_ac.process_net_data import parse_ieee_mat, System, Branch
from power_flow_ac.process_measurements import load_legacy_measurements
from power_flow_ac.compose_meausrement import Pf, Qf, Pi, Qi, Cm, Vm
from power_flow_ac.H_AC import H_AC
from GN_se import GN_se


file = 'nets/ieee118_186.mat'

data = parse_ieee_mat(file)
system_data = data['data']['system']
sys = System(system_data)

legacy_data = data['data']['legacy']
z, v, meas_indexes = load_legacy_measurements(legacy_data)
Pf_idx, Qf_idx, Cm_idx, Pi_idx, Qi_idx, Vm_idx = meas_indexes
branch = Branch(sys.branch)
pi = Pi(Pi_idx, sys.bus, sys.Ybus, sys.Yij, sys.Yii)
qi = Qi(Qi_idx, sys.bus, sys.Ybus, sys.Yij, sys.Yii)
pf = Pf(Pf_idx, branch)
qf = Qf(Qf_idx, branch)
cm = Cm(Cm_idx, sys.bus, branch)
vm = Vm(Vm_idx, sys.bus)

h_ac = H_AC(pi, qi, pf, qf, cm, vm, sys.nb)

tol = 1e-8
max_iter = 500
T_est, V_est, eps = GN_se(z, v, sys.slk_bus, h_ac, sys.nb, tol, max_iter)

pmu = data['data'].get('pmu')
if pmu:
    V_true, T_true = pmu['voltage'][:, -2:].T
    sv_true = np.r_[V_true, T_true]
    sv_est = np.r_[V_est, T_est]
    dx = sv_est - sv_true
    print('**Errors**')
    print(f'Voltage error GN: {np.linalg.norm(V_est - V_true):.4f}')
    print(f'Theta (radians) error GN: {np.linalg.norm(T_est - T_true):.4f}')
    print(f'RMSE: {np.sqrt(np.sum(dx)**2/(2 * sys.nb)):.4f}')
    print(f'eps: {eps}')
    print()
