import numpy as np
import matplotlib.pyplot as plt

from SE_np.optimizers.GN_se import GN_se
from SE_np.net_preprocess.process_net_data import parse_ieee_mat, System, Branch
from SE_np.net_preprocess.process_measurements import load_legacy_measurements
from SE_np.PF_equations.PF_cartesian import H_AC as H_AC_cartesian
from SE_np.PF_equations.PF_polar import H_AC as H_AC_polar
from SE_np.optimizers.SGD_se import FGD_se, step_size
from SE_np.utils import (init_start_point, aggregate_meas_idx, square_mag,
                         sample_from_SGD, RMSE, normalize_measurements, iterative_err)

file = '../../../nets/ieee118_186.mat'

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

sv_true = np.r_[V_true, T_true]

T0, V0 = init_start_point(sys, data, how='flat')
u0 = V0 * np.exp(1j * T0)

measurements_square, h_ac_cart.H, norm_H = normalize_measurements(h_ac_cart.H, measurements_square)
eta = step_size(u0, h_ac_cart.H, measurements_square, norm_H)

u_fgd, all_u_fgd, converged_fgd, k_fgd = FGD_se(u0, measurements_square, h_ac_cart.H, eta, norm_H, sys.slk_bus)
u_fgd, T_fgd, V_fgd, err_fgd = sample_from_SGD(u_fgd, h_ac_cart.H, measurements_square)
x_fgd = np.r_[T_fgd, V_fgd]

x_fgd_gn, all_x_fgd_gn, fgd_gn_converged, k_fgd_gn = GN_se(x_fgd, measurements, variance, sys.slk_bus, h_ac_polar, sys.nb, prefix=' FGD')
T_fgd_gn, V_fgd_gn = x_fgd_gn[:sys.nb], x_fgd_gn[sys.nb:]


u_agd, all_u_agd, converged_agd, k_agd = FGD_se(u0, measurements_square, h_ac_cart.H, eta, norm_H, sys.slk_bus, AGD_update=True)
u_agd, T_agd, V_agd, err_agd = sample_from_SGD(u_agd, h_ac_cart.H, measurements_square)
x_agd = np.r_[T_agd, V_agd]

x_agd_gn, all_x_agd_gn, agd_gn_converged, k_agd_gn = GN_se(x_agd, measurements, variance, sys.slk_bus, h_ac_polar, sys.nb, prefix=' AGD')
T_agd_gn, V_agd_gn = x_agd_gn[:sys.nb], x_agd_gn[sys.nb:]

x0 = np.r_[T0, V0]
x_gn, all_x_gn, gn_converged, k_gn = GN_se(x0, measurements, variance, sys.slk_bus, h_ac_polar, sys.nb)
T_gn, V_gn = x_gn[:sys.nb], x_gn[sys.nb:]


RMSE_fgd = np.real(RMSE(T_true, V_true, T_fgd, V_fgd))
iterative_err_fgd = np.real(iterative_err(T_true, V_true, T_fgd, V_fgd))

RMSE_fgd_gn = RMSE(T_true, V_true, T_fgd_gn, V_fgd_gn)
iterative_err_fgd_gn = np.real(iterative_err(T_true, V_true, T_fgd_gn, V_fgd_gn))

RMSE_agd = np.real(RMSE(T_true, V_true, T_agd, V_agd))
iterative_err_agd = np.real(iterative_err(T_true, V_true, T_agd, V_agd))

RMSE_agd_gn = RMSE(T_true, V_true, T_agd_gn, V_agd_gn)
iterative_err_agd_gn = np.real(iterative_err(T_true, V_true, T_agd_gn, V_agd_gn))

RMSE_gn = RMSE(T_true, V_true, T_gn, V_gn)
iterative_err_gn = np.real(iterative_err(T_true, V_true, T_gn, V_gn))

all_fgd_err = [np.real(iterative_err(T_true, V_true, u_est=u)) for u in all_u_fgd]
all_agd_err = [np.real(iterative_err(T_true, V_true, u_est=u)) for u in all_u_agd]

print(f"[FGD]       RMSE: {RMSE_fgd:.4e}, Iterative ERR: {iterative_err_fgd:.4e}, Steps: {k_fgd}")
print(f"[FGD GN]    RMSE: {RMSE_fgd_gn:.4e}, Iterative ERR: {iterative_err_fgd_gn:.4e}")
print()
print(f"[AGD]       RMSE: {RMSE_agd:.4e}, Iterative ERR: {iterative_err_agd:.4e}, Steps: {k_agd}")
print(f"[AGD GN]    RMSE: {RMSE_agd_gn:.4e}, Iterative ERR: {iterative_err_agd_gn:.4e}")
print()
print(f"[GN]        RMSE: {RMSE_gn:.4e}, Iterative ERR: {iterative_err_gn:.4e}, Steps: {k_gn}")

plt.plot(all_fgd_err, ls='--', label='FGD - 118-bus')
plt.plot(all_agd_err, ls='-.', label='AGD - 118-bus')

plt.xlabel('Number of Iterations')
plt.ylabel(r'$\frac{\|V_1 - V\|_F}{\|V\|_F}$')

x_max = max(len(all_fgd_err), len(all_agd_err))
plt.xticks(np.arange(0, x_max + 1, 100))

y_min = 0
y_max = max(np.max(all_fgd_err), np.max(all_agd_err))
plt.yticks(np.arange(y_min, y_max + 0.1, 0.1))

plt.grid(True, ls=':', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()