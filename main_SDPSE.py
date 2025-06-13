import numpy as np

from power_flow_ac.process_net_data import parse_ieee_mat, System, Branch
from power_flow_ac.process_measurements import load_legacy_measurements
from power_flow_ac.compose_meausrement import Pf, Qf, Pi, Qi, Vm
from power_flow_ac.power_flow_cartesian import H_AC as H_AC_cartesian
from power_flow_ac.power_flow_polar import H_AC as H_AC_polar
from SDP_se import SDP_se
from GN_se import GN_se
from power_flow_ac.init_starting_point import init_start_point

file = 'nets/ieee30_41.mat'

data = parse_ieee_mat(file)
system_data = data['data']['system']
sys = System(system_data)

legacy_data = data['data']['legacy']
pmu = data['data'].get('pmu')
z, v, meas_indexes = load_legacy_measurements(legacy_data,
                                              exact=True, sample=False,
                                              measurement_type=['flow', 'injection', 'voltage', 'current'])
branch = Branch(sys.branch)

h_ac_cart = H_AC_cartesian(sys, branch, meas_indexes)
z[-len(h_ac_cart.vm.i):] = z[-len(h_ac_cart.vm.i):] ** 2
z[len(h_ac_cart.pf.i) + len(h_ac_cart.qf.i): len(h_ac_cart.pf.i) + len(h_ac_cart.qf.i) + len(h_ac_cart.cm.i)] = z[len(h_ac_cart.pf.i) + len(h_ac_cart.qf.i): len(h_ac_cart.pf.i) + len(h_ac_cart.qf.i) + len(h_ac_cart.cm.i)] ** 2
v[-len(h_ac_cart.vm.i):] = v[-len(h_ac_cart.vm.i):] ** 2
v[len(h_ac_cart.pf.i) + len(h_ac_cart.qf.i): len(h_ac_cart.pf.i) + len(h_ac_cart.qf.i) + len(h_ac_cart.cm.i)] = v[len(h_ac_cart.pf.i) + len(h_ac_cart.qf.i): len(h_ac_cart.pf.i) + len(h_ac_cart.qf.i) + len(h_ac_cart.cm.i)] ** 2

V_true, T_true = pmu['voltage'][:, -2:].T
sv_true = np.r_[V_true, T_true]

V_est, V_opt = SDP_se(z, v, sys.slk_bus, h_ac_cart, sys.nb)
V_real, V_imag = np.real(V_est), np.imag(V_est)
T_est = np.arctan2(V_imag, V_real)
V_est = np.sqrt(V_real ** 2 + V_imag ** 2)
sv_est = np.r_[V_est, T_est]
dx = sv_est - sv_true
best_fit = np.sqrt(np.sum(dx ** 2) / (2 * sys.nb))

print(f'RMSE SDR: {best_fit:.4e}')

x0 = np.r_[T_est, V_est]
tol = 1e-10
max_iter = 500
z[-len(h_ac_cart.vm.i):] = np.sqrt(z[-len(h_ac_cart.vm.i):])
v[-len(h_ac_cart.vm.i):] = np.sqrt(v[-len(h_ac_cart.vm.i):])
z[len(h_ac_cart.pf.i) + len(h_ac_cart.qf.i): len(h_ac_cart.pf.i) + len(h_ac_cart.qf.i) + len(h_ac_cart.cm.i)] = np.sqrt(z[len(h_ac_cart.pf.i) + len(h_ac_cart.qf.i): len(h_ac_cart.pf.i) + len(h_ac_cart.qf.i) + len(h_ac_cart.cm.i)])
v[len(h_ac_cart.pf.i) + len(h_ac_cart.qf.i): len(h_ac_cart.pf.i) + len(h_ac_cart.qf.i) + len(h_ac_cart.cm.i)] = np.sqrt(v[len(h_ac_cart.pf.i) + len(h_ac_cart.qf.i): len(h_ac_cart.pf.i) + len(h_ac_cart.qf.i) + len(h_ac_cart.cm.i)])
h_ac_polar = H_AC_polar(sys, branch, meas_indexes)

T_est, V_est, eps, iter, converged = GN_se(z, v, sys.slk_bus, h_ac_polar, sys.nb, tol, max_iter, x0=x0)


sv_est = np.r_[V_est, T_est]
dx = sv_est - sv_true

rmse_wls_sdr = np.sqrt(np.sum(dx ** 2) / (2 * sys.nb))

print(f'RMSE WLS/SDR: {rmse_wls_sdr:.4e}')

T_est, V_est, eps, iter, converged = GN_se(z, v, sys.slk_bus, h_ac_polar, sys.nb, tol, max_iter)

sv_est = np.r_[V_est, T_est]
dx = sv_est - sv_true

rmse_wls_flat = np.sqrt(np.sum(dx ** 2) / (2 * sys.nb))

print(f'RMSE WLS/FLAT: {rmse_wls_flat:.4e}')

best_fit = min(rmse_wls_sdr, rmse_wls_flat)

T_best = T_est
V_best = V_est
for _ in range(50):
    real_imag = np.random.multivariate_normal(np.zeros(sys.nb * 2), .5 * np.block([[np.real(V_opt), -np.imag(V_opt)],
                                                                               [np.imag(V_opt), np.real(V_opt)]]))
    v_rand = real_imag[:sys.nb] + 1j * real_imag[sys.nb:]
    V_real, V_imag = np.real(v_rand), np.imag(v_rand)
    T = np.arctan2(V_imag, V_real)
    V = np.sqrt(V_real ** 2 + V_imag ** 2)
    x0 = np.r_[T, V]
    T_est2, V_est2, eps, iter, converged = GN_se(z, v, sys.slk_bus, h_ac_polar, sys.nb, tol, max_iter, x0=x0)
    sv_est2 = np.r_[V_est2, T_est2]
    dx2 = sv_est2 - sv_true
    # Evaluate fit: sum of squared residuals

    fit = np.sqrt(np.sum(dx2 ** 2) / (2 * sys.nb))

    if fit < best_fit:
        best_fit = fit
        T_best = T_est2
        V_best = V_est2
dx2 = np.r_[V_best, T_best] - sv_true
print(f'RMSE WLS/SDRSampled: {np.sqrt(np.sum(dx2 ** 2) / (2 * sys.nb)):.4e}')
print()
