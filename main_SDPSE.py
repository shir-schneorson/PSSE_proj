import numpy as np

from power_flow_ac.process_net_data import parse_ieee_mat, System, Branch
from power_flow_ac.process_measurements import load_legacy_measurements
from power_flow_ac.compose_meausrement import Pf, Qf, Pi, Qi, Vm
from power_flow_ac.power_flow_cartesian import H_AC
from SDP_se import SDP_se
from power_flow_ac.init_starting_point import init_start_point

file = 'nets/ieee14_20.mat'

data = parse_ieee_mat(file)
system_data = data['data']['system']
sys = System(system_data)

legacy_data = data['data']['legacy']
z, v, meas_indexes = load_legacy_measurements(legacy_data,
                                              exact=True, sample=False,
                                              measurement_type=['flow', 'injection', 'voltage'])
Pf_idx, Qf_idx, Pi_idx, Qi_idx, Vm_idx = meas_indexes
branch = Branch(sys.branch)
pi = Pi(Pi_idx, sys.bus, sys.Ybus, sys.Yij, sys.Yii)
qi = Qi(Qi_idx, sys.bus, sys.Ybus, sys.Yij, sys.Yii)
pf = Pf(Pf_idx, branch)
qf = Qf(Qf_idx, branch)
vm = Vm(Vm_idx, sys.bus)

h_ac = H_AC(pi, qi, pf, qf, vm, sys.nb)
h_ac.H
x0 = np.r_[init_start_point(sys, data, how='flat')]
T = x0[:sys.nb]
Vm = x0[sys.nb:]
Vc = Vm * np.cos(T) + 1j * Vm * np.sin(T)
z_est = h_ac.estimate(Vc)
v_leading, v_best = SDP_se(z, v, sys.slk_bus, h_ac, sys.nb, x0)

v_real, v_imag = np.real(v_leading), np.imag(v_leading)
T_est_leading = np.arctan2(v_imag, v_real)
V_est_leading = np.sqrt(v_real ** 2 + v_imag ** 2)

v_real, v_imag = np.real(v_best), np.imag(v_best)
T_est_best = np.arctan2(v_imag, v_real)
V_est_best = np.sqrt(v_real ** 2 + v_imag ** 2)

pmu = data['data'].get('pmu')
if pmu:
    V_true, T_true = pmu['voltage'][:, -2:].T
    sv_true = np.r_[V_true, T_true]
    sv_est_lead = np.r_[V_est_leading, T_est_leading]
    sv_est_best = np.r_[V_est_best, T_est_best]
    dx_lead = sv_est_lead - sv_true
    dx_best = sv_est_best - sv_true
    print(f'RMSE Lead: {np.sqrt(np.sum(dx_lead ** 2) / (2 * sys.nb)):.4e}')
    print(f'RMSE Best: {np.sqrt(np.sum(dx_best ** 2) / (2 * sys.nb)):.4e}')
    print()
