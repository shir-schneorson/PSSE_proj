import numpy as np
import matplotlib.pyplot as plt

from optimizers.GN_se import GN_se
from power_flow_ac.process_net_data import parse_ieee_mat, System, Branch
from power_flow_ac.process_measurements import load_legacy_measurements
from power_flow_ac.power_flow_cartesian import H_AC as H_AC_cartesian
from power_flow_ac.power_flow_polar import H_AC as H_AC_polar
from optimizers.LM_se import LMOptimizerSE
from utils import aggregate_meas_idx, square_mag, RMSE, normalize_measurements, iterative_err
from power_flow_ac.init_starting_point import init_start_point

file = '../nets/ieee118_186.mat'

data = parse_ieee_mat(file)
system_data = data['data']['system']
sys = System(system_data)
branch = Branch(sys.branch)

pmu = data['data'].get('pmu')

legacy_data = data['data']['legacy']
measurements, variance, meas_indexes = load_legacy_measurements(legacy_data,
                                              exact=False, sample=False,
                                              measurement_type=['flow', 'injection', 'voltage'])
meas_types = ['Pf', 'Qf', 'Cm', 'Pi', 'Qi', 'Vm']
agg_meas_idx = aggregate_meas_idx(meas_indexes, meas_types)
measurements_square, variance_square = square_mag(measurements, variance, agg_meas_idx)

h_ac_polar = H_AC_polar(sys, branch, meas_indexes)
h_ac_cart = H_AC_cartesian(sys, branch, meas_indexes)
V_true, T_true = pmu['voltage'][:, -2:].T

measurements_square, h_ac_cart.H, norm_H = normalize_measurements(measurements_square, h_ac_cart.H)

sv_true = np.r_[V_true, T_true]

T0, V0 = init_start_point(sys, data, how='flat')
u0 = V0 * np.exp(1j * T0)
x0 = np.r_[T0, V0]

# LM_opt = LMOptimizerSE(h_ac_cart.H, measurements_square, sys.slk_bus)
# u_lm, all_u_lm, converged_lm = LM_opt.optimize(u0)
LM_opt = LMOptimizerSE(h_ac_polar, measurements, variance, sys.slk_bus, sys.nb, norm_H)
x_lm, all_x_lm, converged_lm = LM_opt.optimize(x0)
T_lm, V_lm = x_lm[:sys.nb], x_lm[sys.nb:]
# u_lm, T_lm, V_lm, err_lm = sample_from_SGD(u_lm, h_ac_cart.H, measurements_square)

x_lm = np.r_[T_lm, V_lm]
T_lm_wls, V_lm_wls, _, _, wls_fgd_converged = GN_se(measurements, variance, sys.slk_bus, h_ac_polar, sys.nb, x0=x_lm)


T_flat, V_flat, _, _, wls_converged = GN_se(measurements, variance, sys.slk_bus, h_ac_polar, sys.nb, x0=x0)

RMSE_lm = np.real(RMSE(T_true, V_true, T_lm, V_lm, sys.nb))
iterative_err_lm = np.real(iterative_err(T_true, V_true, T_lm, V_lm))

RMSE_lm_wls = RMSE(T_true, V_true, T_lm_wls, V_lm_wls, sys.nb)
iterative_err_lm_wls = np.real(iterative_err(T_true, V_true, T_lm_wls, V_lm_wls))

RMSE_flat_wls = RMSE(T_true, V_true, T_flat, V_flat, sys.nb)
iterative_err_flat_wls = np.real(iterative_err(T_true, V_true, T_flat, V_flat))

all_lm_err = [np.real(iterative_err(T_true, V_true, x[:sys.nb], x[sys.nb:])) for x in all_x_lm]

print(f"[LM]         RMSE: {RMSE_lm:.4e}, Iterative ERR: {iterative_err_lm:.4e}")
print(f"[LM WLS]     RMSE: {RMSE_lm_wls:.4e}, Iterative ERR: {iterative_err_lm_wls:.4e}")
print(f"[FLAT WLS]   RMSE: {RMSE_flat_wls:.4e}, Iterative ERR: {iterative_err_flat_wls:.4e}")

plt.plot(all_lm_err, ls='--', label='FGD - 118-bus')

plt.xlabel('Number of Iterations')
plt.ylabel(r'$\frac{\|V_1 - V\|_F}{\|V\|_F}$')

plt.grid(True, ls=':', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()