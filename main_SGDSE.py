import numpy as np
import matplotlib.pyplot as plt

from GN_se import GN_se
from power_flow_ac.process_net_data import parse_ieee_mat, System, Branch
from power_flow_ac.process_measurements import load_legacy_measurements
from power_flow_ac.power_flow_cartesian import H_AC as H_AC_cartesian
from power_flow_ac.power_flow_polar import H_AC as H_AC_polar
from SGD_se import FGD_se
from utils import aggregate_meas_idx, square_mag, sample_from_SGD, RMSE, normalize_measurements, iterative_err
from power_flow_ac.init_starting_point import init_start_point

file = 'nets/ieee118_186.mat'

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

measurements_square, h_ac_cart.H = normalize_measurements(measurements_square, h_ac_cart.H)

sv_true = np.r_[V_true, T_true]

T0, V0 = init_start_point(sys, data, how='flat')
u0 = V0 * np.exp(1j * T0)

all_u_fgd, converged_fgd = FGD_se(u0, measurements_square, h_ac_cart)
all_u_agd, converged_age = FGD_se(u0, measurements_square, h_ac_cart, AGD_update=True)
u_fgd = all_u_fgd[-1]
u_agd = all_u_agd[-1]

u_fgd, T_fgd, V_fgd, err_fgd = sample_from_SGD(u_fgd, h_ac_cart.H, measurements_square)
u_agd, T_agd, V_agd, err_agd = sample_from_SGD(u_agd, h_ac_cart.H, measurements_square)

x_fgd = np.r_[T_fgd, V_fgd]
T_fgd_wls, V_fgd_wls, _, _, wls_fgd_converged = GN_se(measurements, variance, sys.slk_bus, h_ac_polar, sys.nb, x0=x_fgd)

x_agd = np.r_[T_agd, V_agd]
T_agd_wls, V_agd_wls, _, _, wls_agd_converged = GN_se(measurements, variance, sys.slk_bus, h_ac_polar, sys.nb, x0=x_agd)

x_flat = np.r_[T0, V0]
T_flat, V_flat, _, _, wls_converged = GN_se(measurements, variance, sys.slk_bus, h_ac_polar, sys.nb, x0=x_flat)


RMSE_fgd = np.real(RMSE(T_true, V_true, T_fgd, V_fgd, sys.nb))
iterative_err_fgd = np.real(iterative_err(T_true, V_true, T_fgd, V_fgd))

RMSE_fgd_wls = RMSE(T_true, V_true, T_fgd_wls, V_fgd_wls, sys.nb)
iterative_err_fgd_wls = np.real(iterative_err(T_true, V_true, T_fgd_wls, V_fgd_wls))

RMSE_agd = np.real(RMSE(T_true, V_true, T_agd, V_agd, sys.nb))
iterative_err_agd = np.real(iterative_err(T_true, V_true, T_agd, V_agd))

RMSE_agd_wls = RMSE(T_true, V_true, T_agd_wls, V_agd_wls, sys.nb)
iterative_err_agd_wls = np.real(iterative_err(T_true, V_true, T_agd_wls, V_agd_wls))

RMSE_flat_wls = RMSE(T_true, V_true, T_flat, V_flat, sys.nb)
iterative_err_flat_wls = np.real(iterative_err(T_true, V_true, T_flat, V_flat))

all_fgd_err = [np.real(iterative_err(T_true, V_true, u_est=u)) for u in all_u_fgd]
all_agd_err = [np.real(iterative_err(T_true, V_true, u_est=u)) for u in all_u_agd]

print(f"[FGD]        RMSE: {RMSE_fgd:.4e}, Iterative ERR: {iterative_err_fgd:.4e}")
print(f"[FGD WLS]    RMSE: {RMSE_fgd_wls:.4e}, Iterative ERR: {iterative_err_fgd_wls:.4e}")
print(f"[AGD]        RMSE: {RMSE_agd:.4e}, Iterative ERR: {iterative_err_agd:.4e}")
print(f"[AGD WLS]    RMSE: {RMSE_agd_wls:.4e}, Iterative ERR: {iterative_err_agd_wls:.4e}")
print(f"[FLAT WLS]   RMSE: {RMSE_flat_wls:.4e}, Iterative ERR: {iterative_err_flat_wls:.4e}")

plt.plot(all_fgd_err, ls='--', label='FGD - 118-bus')
plt.plot(all_agd_err, ls='-.', label='AGD - 118-bus')

plt.xlabel('Number of Iterations')
plt.ylabel(r'$\frac{\|V_1 - V\|_F}{\|V\|_F}$')

# Set custom ticks
x_max = max(len(all_fgd_err), len(all_agd_err))
plt.xticks(np.arange(0, x_max + 1, 100))  # Tick every 100 on x-axis

y_min = 0
y_max = max(np.max(all_fgd_err), np.max(all_agd_err))
plt.yticks(np.arange(y_min, y_max + 0.1, 0.1))  # Tick every 0.1 on y-axis

plt.grid(True, ls=':', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()