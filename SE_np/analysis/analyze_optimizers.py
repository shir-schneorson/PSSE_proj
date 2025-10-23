import numpy as np
import matplotlib.pyplot as plt

from SE_np.data_parser.data_generator import DataGenerator
from SE_np.optimizers.LM_se_with_prior import LMOptimizerSE
from SE_np.optimizers.SGD_se import FGD_se, step_size
from SE_np.net_preprocess.process_net_data import parse_ieee_mat, System, Branch
from SE_np.optimizers import GN_se
from SE_np.utils import init_start_point
from SE_np.PF_equations.PF_cartesian import H_AC
from SE_np.utils import generate_data, square_mag, sample_from_SGD, normalize_measurements, iterative_err, RMSE

file = "../../nets/ieee118_186.mat"


def run_experiment(data, sys, branch, init_params, **kwargs):
    iterative_err_fgd = []
    iterative_err_agd = []
    iterative_err_lm = []
    iterative_err_lm_wp = []
    iterative_err_lm_wp_GST = []
    iterative_err_lm_norm = []
    iterative_err_gn = []
    iterative_err_gn_norm = []

    rmse_fgd, rmse_fgd_gn = [], []
    rmse_agd, rmse_agd_gn = [], []
    rmse_lm = []
    rmse_lm_wp, rmse_lm_wp_GST = [], []
    rmse_lm_norm, rmse_lm_norm_gn = [], []
    rmse_gn = []
    rmse_gn_norm, rmse_gn_norm_gn = [], []

    steps_fgd, steps_agd = [], []
    steps_lm, steps_lm_wp, steps_lm_wp_GST, steps_lm_norm = [], [], [], []
    steps_gn, steps_gn_norm = [], []

    T0, V0 = init_start_point(sys, data, how='flat')
    m_GST = np.r_[T0, V0]
    Q_GST = np.linalg.pinv(-np.block([[np.imag(sys.Ybus), np.zeros((sys.nb, sys.nb))],
                  [np.zeros((sys.nb, sys.nb)), np.imag(sys.Ybus)]]))
    # Q = np.diag(np.r_[np.ones(sys.nb) * np.pi * init_params[0] / 3, np.ones(sys.nb) * .05 / 3] ** 2)
    data_generator = kwargs.get('data_generator', None)
    m = data_generator.m if data_generator is not None else m_GST
    Q = data_generator.cov if data_generator is not None else np.diag(np.r_[np.ones(sys.nb) * np.pi * init_params[0] / 3, np.ones(sys.nb) * .05 / 3] ** 2)

    for i in range(100):
        gen_data = generate_data(data, sys, branch, init_params, **kwargs)
        measurements, variance, meas_idx, agg_meas_idx, h_ac_cart, h_ac_polar, T_true, V_true, Vc_true = gen_data
        measurements_square, variance_square = square_mag(measurements, variance, agg_meas_idx)
        h_ac_cart_norm = H_AC(sys, branch, meas_idx)
        measurements_square_norm, h_ac_cart_norm.H, norm_H = normalize_measurements(h_ac_cart.H, measurements_square)

        T0, V0 = init_start_point(sys, data, how='flat', dc_init=(branch, measurements, variance, meas_idx, agg_meas_idx))
        x0 = np.r_[T0, V0]

        u0 = V0 * np.exp(1j * T0)
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

        LM_opt = LMOptimizerSE(h_ac_polar, measurements, variance, sys.slk_bus, sys.nb, eta)
        x_lm, all_x_lm, lm_converged, k_lm = LM_opt.optimize(x0)
        T_lm, V_lm = x_lm[:sys.nb], x_lm[sys.nb:]

        ################################## Prior ##################################
        reg_scale = 0
        LM_wp_opt = LMOptimizerSE(h_ac_polar, measurements, variance, sys.slk_bus, sys.nb, eta, m=m, Q=Q,
                                  reg_scale=reg_scale, prefix=f' - with prior (scale {reg_scale})')
        x_lm_wp, all_x_lm_wp, lm_wp_converged, k_lm_wp = LM_wp_opt.optimize(x0)
        T_lm_wp, V_lm_wp = x_lm_wp[:sys.nb], x_lm_wp[sys.nb:]
        RMSE_lm_wp = RMSE(T_true, V_true, T_lm_wp, V_lm_wp)
        prev_reg_scale = reg_scale
        reg_scale = 1

        while np.abs(reg_scale - prev_reg_scale) > 1e-2:
            LM_wp_opt.set_reg_scale(reg_scale)
            LM_wp_opt.prefix = f' - with prior (scale {reg_scale:.7e})'
            x_lm_wp_curr, all_x_lm_wp_curr, lm_wp_converged_curr, k_lm_wp_curr = LM_wp_opt.optimize(x0)
            T_lm_wp_curr, V_lm_wp_curr = x_lm_wp_curr[:sys.nb], x_lm_wp_curr[sys.nb:]
            RMSE_lm_wp_curr = RMSE(T_true, V_true, T_lm_wp_curr, V_lm_wp_curr)
            step = np.abs(reg_scale - prev_reg_scale) / 2
            if RMSE_lm_wp_curr <= RMSE_lm_wp:
                reg_scale += step
                prev_reg_scale = reg_scale
                RMSE_lm_wp = RMSE_lm_wp_curr
                x_lm_wp, T_lm_wp, V_lm_wp = x_lm_wp_curr, T_lm_wp_curr, V_lm_wp_curr
                all_x_lm_wp, lm_wp_converged, k_lm_wp = all_x_lm_wp_curr, lm_wp_converged_curr, k_lm_wp_curr
            if RMSE_lm_wp_curr > RMSE_lm_wp:
                reg_scale -= step

        reg_scale_GST = 0
        LM_wp_GST_opt = LMOptimizerSE(h_ac_polar, measurements, variance, sys.slk_bus, sys.nb, eta, m=m_GST, Q=Q_GST,
                                  reg_scale=reg_scale_GST, prefix=f' - with prior GST (scale {reg_scale_GST})')
        x_lm_wp_GST, all_x_lm_wp_GST, lm_wp_GST_converged, k_lm_wp_GST = LM_wp_GST_opt.optimize(x0)
        T_lm_wp_GST, V_lm_wp_GST = x_lm_wp_GST[:sys.nb], x_lm_wp_GST[sys.nb:]
        RMSE_lm_wp_GST = RMSE(T_true, V_true, T_lm_wp_GST, V_lm_wp_GST)

        prev_reg_scale_GST = reg_scale_GST
        reg_scale_GST = 1
        while np.abs(reg_scale_GST - prev_reg_scale_GST) > 1e-2:
            LM_wp_GST_opt.set_reg_scale(reg_scale_GST)
            LM_wp_GST_opt.prefix = f' - with prior GST (scale {reg_scale_GST:.7e})'
            x_lm_wp_GST_curr, all_x_lm_wp_GST_curr, lm_wp_GST_converged_curr, k_lm_wp_GST_curr = LM_wp_GST_opt.optimize(x0)
            T_lm_wp_GST_curr, V_lm_wp_GST_curr = x_lm_wp_GST_curr[:sys.nb], x_lm_wp_GST_curr[sys.nb:]
            RMSE_lm_wp_GST_curr = RMSE(T_true, V_true, T_lm_wp_GST_curr, V_lm_wp_GST_curr)
            step = np.abs(reg_scale_GST - prev_reg_scale_GST) / 2
            if RMSE_lm_wp_GST_curr <= RMSE_lm_wp_GST:
                reg_scale_GST += step
                prev_reg_scale_GST = reg_scale_GST
                RMSE_lm_wp_GST = RMSE_lm_wp_GST_curr
                x_lm_wp_GST, T_lm_wp_GST, V_lm_wp_GST = x_lm_wp_GST_curr, T_lm_wp_GST_curr, V_lm_wp_GST_curr
                all_x_lm_wp_GST, lm_wp_GST_converged, k_lm_wp_GST = all_x_lm_wp_GST_curr, lm_wp_GST_converged_curr, k_lm_wp_GST_curr
            if RMSE_lm_wp_GST_curr > RMSE_lm_wp_GST:
                reg_scale_GST -= step

        ###########################################################################

        LM_norm_opt = LMOptimizerSE(h_ac_polar, measurements, variance, sys.slk_bus, sys.nb, eta, norm_H=norm_H)
        x_lm_norm, all_x_lm_norm, lm_norm_converged, k_lm_norm = LM_norm_opt.optimize(x0)
        T_lm_norm, V_lm_norm = x_lm_norm[:sys.nb], x_lm_norm[sys.nb:]
        x_lm_norm_gn, all_x_lm_norm_gn, lm_norm_gn_converged, k_lm_norm_gn = GN_se(x_lm_norm, measurements, variance, sys.slk_bus, h_ac_polar, sys.nb, prefix=' LM-norm')
        T_lm_norm_gn, V_lm_norm_gn = x_lm_norm_gn[:sys.nb], x_lm_norm_gn[sys.nb:]

        x_gn, all_x_gn, gn_converged, k_gn = GN_se(x0, measurements, variance, sys.slk_bus, h_ac_polar, sys.nb)
        T_gn, V_gn = x_gn[:sys.nb], x_gn[sys.nb:]

        x_gn_norm, all_x_gn_norm, gn_norm_converged, k_gn_norm = GN_se(x0, measurements, variance, sys.slk_bus, h_ac_polar, sys.nb, norm_H=norm_H)
        T_gn_norm, V_gn_norm = x_gn_norm[:sys.nb], x_gn_norm[sys.nb:]
        x_gn_norm_gn, all_x_gn_norm_gn, gn_norm_gn_converged, k_gn_norm_gn = GN_se(x_gn_norm, measurements, variance, sys.slk_bus, h_ac_polar, sys.nb, prefix=' GN-norm')
        T_gn_norm_gn, V_gn_norm_gn = x_gn_norm_gn[:sys.nb], x_gn_norm_gn[sys.nb:]

        rmse_gn.append(RMSE(T_true, V_true, T_gn, V_gn))
        rmse_gn_norm.append(RMSE(T_true, V_true, T_gn_norm, V_gn_norm))
        rmse_gn_norm_gn.append(RMSE(T_true, V_true, T_gn_norm_gn, V_gn_norm_gn))

        rmse_fgd.append(RMSE(T_true, V_true, T_fgd, V_fgd))
        rmse_fgd_gn.append(RMSE(T_true, V_true, T_fgd_gn, V_fgd_gn))

        rmse_agd.append(RMSE(T_true, V_true, T_agd, V_agd))
        rmse_agd_gn.append(RMSE(T_true, V_true, T_agd_gn, V_agd_gn))

        rmse_lm.append(RMSE(T_true, V_true, T_lm, V_lm))
        rmse_lm_wp.append(RMSE(T_true, V_true, T_lm_wp, V_lm_wp))
        rmse_lm_wp_GST.append(RMSE(T_true, V_true, T_lm_wp_GST, V_lm_wp_GST))

        rmse_lm_norm.append(RMSE(T_true, V_true, T_lm_norm, V_lm_norm))
        rmse_lm_norm_gn.append(RMSE(T_true, V_true, T_lm_norm_gn, V_lm_norm_gn))


        all_iterative_err_fgd = np.r_[[np.real(iterative_err(T_true, V_true, u_est=u)) for u in all_u_fgd],
                                        [np.nan] * (501 - len(all_u_fgd))]
        all_iterative_err_agd = np.r_[[np.real(iterative_err(T_true, V_true, u_est=u)) for u in all_u_agd],
                                        [np.nan] * (501 - len(all_u_agd))]
        all_iterative_err_lm = np.r_[[np.real(iterative_err(T_true, V_true, x[:sys.nb], x[sys.nb:])) for x in all_x_lm],
                                        [np.nan] * (501 - len(all_x_lm))]
        all_iterative_err_lm_wp = np.r_[[np.real(iterative_err(T_true, V_true, x[:sys.nb], x[sys.nb:])) for x in all_x_lm_wp],
                                        [np.nan] * (501 - len(all_x_lm_wp))]
        all_iterative_err_lm_wp_GST = np.r_[[np.real(iterative_err(T_true, V_true, x[:sys.nb], x[sys.nb:])) for x in all_x_lm_wp_GST],
                                        [np.nan] * (501 - len(all_x_lm_wp_GST))]
        all_iterative_err_lm_norm = np.r_[[np.real(iterative_err(T_true, V_true, x[:sys.nb], x[sys.nb:])) for x in all_x_lm_norm],
                                        [np.nan] * (501 - len(all_x_lm_norm))]
        all_iterative_err_gn = np.r_[[np.real(iterative_err(T_true, V_true, x[:sys.nb], x[sys.nb:])) for x in all_x_gn],
                                        [np.nan] * (501 - len(all_x_gn))]
        all_iterative_err_gn_norm = np.r_[[np.real(iterative_err(T_true, V_true, x[:sys.nb], x[sys.nb:])) for x in all_x_gn_norm],
                                        [np.nan] * (501 - len(all_x_gn_norm))]

        iterative_err_fgd.append(all_iterative_err_fgd)
        iterative_err_agd.append(all_iterative_err_agd)
        iterative_err_lm.append(all_iterative_err_lm)
        iterative_err_lm_wp.append(all_iterative_err_lm_wp)
        iterative_err_lm_wp_GST.append(all_iterative_err_lm_wp_GST)
        iterative_err_lm_norm.append(all_iterative_err_lm_norm)
        iterative_err_gn.append(all_iterative_err_gn)
        iterative_err_gn_norm.append(all_iterative_err_gn_norm)

        steps_fgd.append(k_fgd + k_fgd_gn)
        steps_agd.append(k_agd + k_agd_gn)
        steps_lm.append(k_lm)
        steps_lm_wp.append(k_lm_wp)
        steps_lm_wp_GST.append(k_lm_wp_GST)
        steps_lm_norm.append(k_lm_norm + k_lm_norm_gn)
        steps_gn.append(k_gn)
        steps_gn_norm.append(k_gn_norm + k_gn_norm_gn)


        if kwargs.get('verbose', False):
            print()
            print(f'[Trial {i + 1}] Converged: {fgd_gn_converged}, RMSE FGD: {rmse_fgd[-1]:.8f}, RMSE FGD GN: {rmse_fgd_gn[-1]:.8f}, steps: {steps_fgd[-1]}')
            print(f'[Trial {i + 1}] Converged: {agd_gn_converged}, RMSE AGD: {rmse_agd[-1]:.8f}, RMSE AGD GN: {rmse_agd_gn[-1]:.8f}, steps: {steps_agd[-1]}')
            print(f'[Trial {i + 1}] Converged: {lm_converged}, RMSE LM: {rmse_lm[-1]:.8f}, steps: {steps_lm[-1]}')
            print(f'[Trial {i + 1}] Converged: {lm_wp_converged}, RMSE LM - with prior (reg scale {prev_reg_scale:.3e}): {rmse_lm_wp[-1]:.8f}, steps: {steps_lm_wp[-1]}')
            print(f'[Trial {i + 1}] Converged: {lm_wp_GST_converged}, RMSE LM - with prior GST (reg scale {prev_reg_scale_GST:.3e}): {rmse_lm_wp_GST[-1]:.8f}, steps: {steps_lm_wp_GST[-1]}')
            print(f'[Trial {i + 1}] Converged: {lm_norm_gn_converged}, RMSE LM norm: {rmse_lm_norm[-1]:.8f}, RMSE LM norm GN: {rmse_lm_norm_gn[-1]:.8f}, steps: {steps_lm_norm[-1]}')
            print(f'[Trial {i + 1}] Converged: {gn_converged}, RMSE GN: {rmse_gn[-1]:.8f}, steps: {steps_gn[-1]}')
            print(f'[Trial {i + 1}] Converged: {gn_norm_converged}, RMSE GN norm: {rmse_gn_norm[-1]:.8f}, RMSE GN norm GN: {rmse_gn_norm_gn[-1]:.8f}, steps: {steps_gn_norm[-1]}')
            print()


    average_iterative_error_fgd = np.nanmean(np.vstack(iterative_err_fgd), axis=0)
    average_iterative_error_agd = np.nanmean(np.vstack(iterative_err_agd), axis=0)
    average_iterative_error_lm = np.nanmean(np.vstack(iterative_err_lm), axis=0)
    average_iterative_error_lm_wp = np.nanmean(np.vstack(iterative_err_lm_wp), axis=0)
    average_iterative_error_lm_wp_GST = np.nanmean(np.vstack(iterative_err_lm_wp_GST), axis=0)
    average_iterative_error_lm_norm = np.nanmean(np.vstack(iterative_err_lm_norm), axis=0)
    average_iterative_error_gn = np.nanmean(np.vstack(iterative_err_gn), axis=0)
    average_iterative_error_gn_norm = np.nanmean(np.vstack(iterative_err_gn_norm), axis=0)

    average_steps_fgd = np.nanmean(steps_fgd)
    average_steps_agd = np.nanmean(steps_agd)
    average_steps_lm = np.nanmean(steps_lm)
    average_steps_lm_wp = np.nanmean(steps_lm_wp)
    average_steps_lm_wp_GST = np.nanmean(steps_lm_wp_GST)
    average_steps_lm_norm = np.nanmean(steps_lm_norm)
    average_steps_gn = np.nanmean(steps_gn)
    average_steps_gn_norm = np.nanmean(steps_gn_norm)


    return (rmse_gn, rmse_gn_norm, rmse_gn_norm_gn,
            rmse_fgd, rmse_fgd_gn,
            rmse_agd, rmse_agd_gn,
            rmse_lm, rmse_lm_wp, rmse_lm_wp_GST, rmse_lm_norm, rmse_lm_norm_gn,
            average_steps_fgd, average_steps_agd, average_steps_lm, average_steps_lm_wp, average_steps_lm_wp_GST, average_steps_lm_norm,
            average_steps_gn, average_steps_gn_norm,
            average_iterative_error_gn, average_iterative_error_gn_norm,
            average_iterative_error_fgd, average_iterative_error_agd,
            average_iterative_error_lm, average_iterative_error_lm_wp, average_iterative_error_lm_wp_GST, average_iterative_error_lm_norm)


def main():
    data = parse_ieee_mat(file)
    system_data = data['data']['system']
    sys = System(system_data)
    branch = Branch(sys.branch)
    init_params= (.35, .95, 1.05)
    data_generator = DataGenerator()
    kwargs = {'flow': True, 'injection': True, 'voltage': True, 'current': False,
              'noise': True, 'Pf_noise': 4e-4, 'Qf_noise': 4e-4, 'Cm_noise': 1e-4,
              'Pi_noise': 16e-4, 'Qi_noise': 16e-4, 'Vm_noise': 1.6e-05, 'verbose': True,
              'data_generator': data_generator}
    (rmse_gn, rmse_gn_norm, rmse_gn_norm_gn,
     rmse_fgd, rmse_fgd_gn,
     rmse_agd, rmse_agd_gn,
     rmse_lm, rmse_lm_wp, rmse_lm_wp_GST, rmse_lm_norm, rmse_lm_norm_gn,
     average_steps_fgd, average_steps_agd, average_steps_lm, average_steps_lm_wp, average_steps_lm_wp_GST, average_steps_lm_norm,
     average_steps_gn, average_steps_gn_norm,
     average_iterative_error_gn, average_iterative_error_gn_norm,
     average_iterative_error_fgd, average_iterative_error_agd,
     average_iterative_error_lm, average_iterative_error_lm_wp, average_iterative_error_lm_wp_GST, average_iterative_error_lm_norm) = \
        run_experiment(data, sys, branch, init_params, **kwargs)

    mean_rmse_gn = np.nanmean(rmse_gn)
    mean_rmse_gn_norm, mean_rmse_gn_norm_gn = np.mean(rmse_gn_norm), np.mean(rmse_gn_norm_gn)
    mean_rmse_fgd, mean_rmse_fgd_gn = np.mean(rmse_fgd), np.mean(rmse_fgd_gn)
    mean_rmse_agd, mean_rmse_agd_gn = np.mean(rmse_agd), np.mean(rmse_agd_gn)
    mean_rmse_lm = np.nanmean(rmse_lm)
    mean_rmse_lm_wp = np.nanmean(rmse_lm_wp)
    mean_rmse_lm_wp_GST = np.nanmean(rmse_lm_wp_GST)
    mean_rmse_lm_norm, mean_rmse_lm_norm_gn = np.mean(rmse_lm_norm), np.mean(rmse_lm_norm_gn)


    print(f'Mean RMSE FGD: {mean_rmse_fgd:.8f}, Mean RMSE FGD-GN: {mean_rmse_fgd_gn:.8f}, Mean steps: {average_steps_fgd}')
    print(f'Mean RMSE AGD: {mean_rmse_agd:.8f}, Mean RMSE AGD-GN: {mean_rmse_agd_gn:.8f}, Mean steps: {average_steps_agd}')
    print(f'Mean RMSE LM: {mean_rmse_lm:.8f}, Mean steps: {average_steps_lm}')
    print(f'Mean RMSE LM - with prior: {mean_rmse_lm_wp:.8f}, Mean steps: {average_steps_lm_wp}')
    print(f'Mean RMSE LM - with prior GST: {mean_rmse_lm_wp_GST:.8f}, Mean steps: {average_steps_lm_wp_GST}')
    print(f'Mean RMSE LM-norm: {mean_rmse_lm_norm:.8f}, Mean RMSE LM-norm-GN: {mean_rmse_lm_norm_gn:.8f}, Mean steps: {average_steps_lm_norm}')
    print(f'Mean RMSE GN: {mean_rmse_gn:.8f}, Mean steps: {average_steps_gn}')
    print(f'Mean RMSE GN-norm: {mean_rmse_gn_norm:.8f}, Mean RMSE GN-norm-GN: {mean_rmse_gn_norm_gn:.8f}, Mean steps: {average_steps_gn_norm}')

    # --- Figure 1: Convergence Curves ---
    fig1 = plt.figure(figsize=(10, 5))
    ax0 = fig1.add_subplot(1, 1, 1)

    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd', '#8c564b', '#3b563a']

    ax0.plot(average_iterative_error_gn, color=colors[0], label='GN', linestyle='-')
    ax0.plot(average_iterative_error_gn_norm, color=colors[1], label='GN-norm', linestyle='-')
    ax0.plot(average_iterative_error_fgd, color=colors[2], label='FGD', linestyle='-.')
    ax0.plot(average_iterative_error_agd, color=colors[3], label='AGD', linestyle=':')
    ax0.plot(average_iterative_error_lm, color=colors[4], label='LM', linestyle='--')
    ax0.plot(average_iterative_error_lm_wp, color=colors[5], label='LM-wp', linestyle='--')
    ax0.plot(average_iterative_error_lm_norm, color=colors[6], label='LM-norm', linestyle='--')

    ax0.set_xlabel('Number of Iterations')
    ax0.set_ylabel(r'$\frac{\|V_1 - V\|_F}{\|V\|_F}$')
    x_max = max(len(average_iterative_error_fgd), len(average_iterative_error_agd))
    ax0.set_xticks(np.arange(0, x_max + 1, 100))

    y_max = max(
        np.nanmax(average_iterative_error_gn),
        np.nanmax(average_iterative_error_gn_norm),
        np.nanmax(average_iterative_error_fgd),
        np.nanmax(average_iterative_error_agd),
        np.nanmax(average_iterative_error_lm),
        np.nanmax(average_iterative_error_lm_wp),
        np.nanmax(average_iterative_error_lm_norm),
    )
    ax0.set_yticks(np.arange(0, y_max + 0.1, 0.1))
    ax0.grid(True, ls=':', alpha=0.7)
    ax0.legend()
    ax0.set_title("Convergence of Iterative Methods")
    plt.tight_layout()

    # --- Figure 2: Histogram of RMSEs ---
    fig2 = plt.figure(figsize=(4, 5))
    ax1 = fig2.add_subplot(1, 1, 1)

    rmse_data = {
        'GN': rmse_gn,
        'GN-norm': rmse_gn_norm_gn,
        'FGD': rmse_fgd_gn,
        'AGD': rmse_agd_gn,
        'LM': rmse_lm,
        'LM - with prior': rmse_lm_wp,
        'LM-norm': rmse_lm_norm_gn,
    }
    import json
    with open('rmse_data.json', 'w') as f:
        json.dump(rmse_data, f)

    ax1.hist(
        rmse_data.values(),
        bins='auto',
        label=rmse_data.keys(),
        color=colors,
        stacked=True,
        orientation='vertical',  # now truly vertical (default)
        alpha=0.8
    )
    ax1.set_xlabel('RMSE')
    ax1.set_ylabel('Frequency')
    ax1.set_title('RMSE Distribution')
    ax1.set_xscale('log')
    ax1.legend(fontsize='small')
    ax1.grid(ls=':', alpha=0.6)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()






