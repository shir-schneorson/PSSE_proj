import numpy as np

from SE_np.optimizers import GN_se
from SE_np.net_preprocess.process_net_data import parse_ieee_mat, System, Branch
from SE_np.net_preprocess.process_measurements import load_legacy_measurements
from SE_np.PF_equations.PF_cartesian import H_AC as H_AC_cartesian
from SE_np.PF_equations.PF_polar import H_AC as H_AC_polar
from SE_np.optimizers.LM_se_with_prior import LMOptimizerSE
from SE_np.optimizers.SGD_se import step_size
from SE_np.utils import aggregate_meas_idx, square_mag, RMSE, normalize_measurements
from SE_np.utils import init_start_point

file = '/nets/ieee118_186.mat'

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
h_ac_cart_norm = H_AC_cartesian(sys, branch, meas_indexes)
V_true, T_true = pmu['voltage'][:, -2:].T

measurements_square_norm, h_ac_cart_norm.H, norm_H = normalize_measurements(h_ac_cart.H, measurements_square)

sv_true = np.r_[V_true, T_true]

T0, V0 = init_start_point(sys, data, how='flat')
u0 = V0 * np.exp(1j * T0)
x0 = np.r_[T0, V0]
eta = step_size(u0, h_ac_cart.H, measurements_square, norm_H)

LM_opt = LMOptimizerSE(h_ac_polar, measurements, variance, sys.slk_bus, sys.nb, eta)

x_lm, all_x_lm, lm_converged, k_lm = LM_opt.optimize(x0)
T_lm, V_lm = x_lm[:sys.nb], x_lm[sys.nb:]
RMSE_lm = np.real(RMSE(T_true, V_true, T_lm, V_lm))

##### Prior #####
m = np.load('../../data_parser/data/time_series/mean.npy')
Q = np.load('../../data_parser/data/time_series/covariance.npy')
Q_inv = np.linalg.pinv(Q)
# mu_T, mu_V = 0.045, 10
# m = np.r_[T0, V0]
# Q = -np.block([[mu_T * np.imag(sys.Ybus), np.zeros((sys.nb, sys.nb))],
#               [np.zeros((sys.nb, sys.nb)), mu_V * np.imag(sys.Ybus)]])
# Q_inv = np.linalg.pinv(Q)

reg_scale = 0
LM_wp_opt = LMOptimizerSE(h_ac_polar, measurements, variance, sys.slk_bus, sys.nb, eta, m=m, Q=Q_inv,
                          reg_scale=reg_scale, prefix=f' - with prior (scale {reg_scale})')
x_lm_wp, all_x_lm_wp, lm_wp_converged, k_lm_wp = LM_wp_opt.optimize(x0)
T_lm_wp, V_lm_wp = x_lm_wp[:sys.nb], x_lm_wp[sys.nb:]
RMSE_lm_wp = RMSE(T_true, V_true, T_lm_wp, V_lm_wp)

prev_reg_scale = reg_scale
reg_scale = 1

while np.abs(reg_scale - prev_reg_scale) > 1e-7:
    LM_wp_opt.set_reg_scale(reg_scale)
    LM_wp_opt.prefix = f' - with prior (scale {reg_scale:.7e})'
    x_lm_wp_curr, all_x_lm_wp_curr, lm_wp_converged_curr, k_lm_wp_curr = LM_wp_opt.optimize(x0)
    T_lm_wp_curr, V_lm_wp_curr = x_lm_wp_curr[:sys.nb], x_lm_wp_curr[sys.nb:]
    RMSE_lm_wp_curr = RMSE(T_true, V_true, T_lm_wp_curr, V_lm_wp_curr)
    step = np.abs(reg_scale - prev_reg_scale) / 2
    if RMSE_lm_wp_curr <= RMSE_lm_wp:
        prev_reg_scale = reg_scale
        reg_scale += step
        RMSE_lm_wp = RMSE_lm_wp_curr
        x_lm_wp, T_lm_wp, V_lm_wp = x_lm_wp_curr, T_lm_wp_curr, V_lm_wp_curr
        all_x_lm_wp, lm_wp_converged, k_lm_wp = all_x_lm_wp_curr, lm_wp_converged_curr, k_lm_wp_curr
    if RMSE_lm_wp_curr > RMSE_lm_wp:
        reg_scale -= step

#################


x_gn, all_x_gn, gn_converged, k_gn = GN_se(x0, measurements, variance, sys.slk_bus, h_ac_polar, sys.nb)
T_gn, V_gn = x_gn[:sys.nb], x_gn[sys.nb:]

RMSE_lm_wp = np.real(RMSE(T_true, V_true, T_lm_wp, V_lm_wp))
RMSE_lm = np.real(RMSE(T_true, V_true, T_lm, V_lm))
RMSE_gn = RMSE(T_true, V_true, T_gn, V_gn)
RMSE_flat = RMSE(T_true, V_true, T0, V0)

print('\nRMSE Results')
print(f"[LM WITH PRIOR (scale {prev_reg_scale:.2f}]      RMSE: {RMSE_lm_wp:.8f}")
print(f"[LM]                             RMSE: {RMSE_lm:.8f}")
print(f"[GN]                             RMSE: {RMSE_gn:.8f}")
print(f"[FLAT INIT POINT]                RMSE: {RMSE_flat:.8f}")

# all_lm_err = [np.real(iterative_err(T_true, V_true, x[:sys.nb], x[sys.nb:])) for x in all_x_lm]
#
# plt.plot(all_lm_err, ls='--', label='FGD - 118-bus')
#
# plt.xlabel('Number of Iterations')
# plt.ylabel(r'$\frac{\|V_1 - V\|_F}{\|V\|_F}$')
#
# plt.grid(True, ls=':', alpha=0.7)
# plt.legend()
# plt.tight_layout()
# plt.show()