import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from tqdm import tqdm

from LM_se import LMOptimizerSE
from SGD_se import FGD_se, step_size
from power_flow_ac.process_net_data import parse_ieee_mat, System, Branch
from power_flow_ac.power_flow_cartesian import H_AC as H_AC_cartesian
from SDP_se import SDP_se
from GN_se import GN_se
from power_flow_ac.init_starting_point import init_start_point
from utils import generate_data, square_mag, sample_from_SGD, RMSE, normalize_measurements, calc_dT, sample_from_SDR, \
    iterative_err

file = "/Users/shirschneorson/PycharmProjects/PSSE_proj/nets/ieee118_186.mat"


def run_experiment(data, sys, branch, init_params, **kwargs):
    iterative_err_fgd = []
    iterative_err_agd = []
    iterative_err_lm = []
    iterative_err_lm_norm = []
    iterative_err_gn = []
    iterative_err_gn_norm = []

    rmse_fgd, rmse_fgd_gn = [], []
    rmse_agd, rmse_agd_gn = [], []
    rmse_lm, rmse_lm_gn = [], []
    rmse_lm_norm, rmse_lm_norm_gn = [], []
    rmse_gn = []
    rmse_gn_norm, rmse_gn_norm_gn = [], []


    for i in range(100):
        gen_data = generate_data(data, sys, branch, init_params, **kwargs)
        measurements, variance, meas_idx, agg_meas_idx, h_ac_cart, h_ac_polar, T_true, V_true, Vc_true = gen_data
        measurements_square, variance_square = square_mag(measurements, variance, agg_meas_idx)
        measurements_square, h_ac_cart.H, norm_H = normalize_measurements(measurements_square, h_ac_cart.H)
        T0, V0 = init_start_point(sys, data, how='flat')
        x0 = np.r_[T0, V0]

        u0 = V0 * np.exp(1j * T0)
        eta = step_size(u0, h_ac_cart.H, measurements_square)

        u_fgd, all_u_fgd, converged_fgd = FGD_se(u0, measurements_square, h_ac_cart, eta)
        u_fgd, T_fgd, V_fgd, err_fgd = sample_from_SGD(u_fgd, h_ac_cart.H, measurements_square)
        x_fgd = np.r_[T_fgd, V_fgd]
        x_fgd_gn, all_x_fgd_gn, fgd_gn_converged = GN_se(x_fgd, measurements, variance, sys.slk_bus, h_ac_polar, sys.nb, prefix='FGD')
        T_fgd_gn, V_fgd_gn = x_fgd_gn[:sys.nb], x_fgd_gn[sys.nb:]

        u_agd, all_u_agd, converged_age = FGD_se(u0, measurements_square, h_ac_cart, eta, AGD_update=True)
        u_agd, T_agd, V_agd, err_agd = sample_from_SGD(u_agd, h_ac_cart.H, measurements_square)
        x_agd = np.r_[T_agd, V_agd]
        x_agd_gn, all_x_agd_gn, agd_gn_converged = GN_se(x_agd, measurements, variance, sys.slk_bus, h_ac_polar, sys.nb, prefix='AGD')
        T_agd_gn, V_agd_gn = x_agd_gn[:sys.nb], x_agd_gn[sys.nb:]

        LM_opt = LMOptimizerSE(h_ac_polar, measurements, variance, sys.slk_bus, sys.nb, eta)
        x_lm, all_x_lm, converged_lm = LM_opt.optimize(x0)
        T_lm, V_lm = x_lm[:sys.nb], x_lm[sys.nb:]
        x_lm_gn, all_x_lm_gn, lm_gn_converged = GN_se(x_lm, measurements, variance, sys.slk_bus, h_ac_polar, sys.nb, prefix='LM')
        T_lm_gn, V_lm_gn = x_lm_gn[:sys.nb], x_lm_gn[sys.nb:]

        LM_norm_opt = LMOptimizerSE(h_ac_polar, measurements, variance, sys.slk_bus, sys.nb, eta, norm_H=norm_H)
        x_lm_norm, all_x_lm_norm, converged_lm_norm = LM_norm_opt.optimize(x0)
        T_lm_norm, V_lm_norm = x_lm_norm[:sys.nb], x_lm_norm[sys.nb:]
        x_lm_norm_gn, all_x_lm_norm_gn, lm_norm_gn_converged = GN_se(x_lm, measurements, variance, sys.slk_bus, h_ac_polar, sys.nb, prefix='LM-norm')
        T_lm_norm_gn, V_lm_norm_gn = x_lm_norm_gn[:sys.nb], x_lm_norm_gn[sys.nb:]

        x_gn, all_x_gn, gn_converged = GN_se(x0, measurements, variance, sys.slk_bus, h_ac_polar, sys.nb)
        T_gn, V_gn = x_gn[:sys.nb], x_gn[sys.nb:]

        x_gn_norm, all_x_gn_norm, gn_norm_converged = GN_se(x0, measurements, variance, sys.slk_bus, h_ac_polar, sys.nb, norm_H=norm_H)
        T_gn_norm, V_gn_norm = x_gn_norm[:sys.nb], x_gn_norm[sys.nb:]
        x_gn_norm_gn, all_x_gn_norm_gn, gn_norm_gn_converged = GN_se(x_gn_norm, measurements, variance, sys.slk_bus, h_ac_polar, sys.nb, prefix='GN-norm')
        T_gn_norm_gn, V_gn_norm_gn = x_gn_norm_gn[:sys.nb], x_gn_norm_gn[sys.nb:]

        rmse_gn.append(iterative_err(T_true, V_true, T_gn, V_gn))
        rmse_gn_norm.append(iterative_err(T_true, V_true, T_gn_norm, V_gn_norm))
        rmse_gn_norm_gn.append(iterative_err(T_true, V_true, T_gn_norm_gn, V_gn_norm_gn))

        rmse_fgd.append(iterative_err(T_true, V_true, T_fgd, V_fgd))
        rmse_fgd_gn.append(iterative_err(T_true, V_true, T_fgd_gn, V_fgd_gn))

        rmse_agd.append(iterative_err(T_true, V_true, T_agd, V_agd))
        rmse_agd_gn.append(iterative_err(T_true, V_true, T_agd_gn, V_agd_gn))

        rmse_lm.append(iterative_err(T_true, V_true, T_lm, V_lm))
        rmse_lm_gn.append(iterative_err(T_true, V_true, T_lm_gn, V_lm_gn))

        rmse_lm_norm.append(iterative_err(T_true, V_true, T_lm_norm, V_lm_norm))
        rmse_lm_norm_gn.append(iterative_err(T_true, V_true, T_lm_norm_gn, V_lm_norm_gn))


        all_iterative_err_fgd = np.r_[[np.real(iterative_err(T_true, V_true, u_est=u)) for u in all_u_fgd],
                                        [np.nan] * (500 - len(all_u_fgd))]
        all_iterative_err_agd = np.r_[[np.real(iterative_err(T_true, V_true, u_est=u)) for u in all_u_agd],
                                        [np.nan] * (500 - len(all_u_agd))]
        all_iterative_err_lm = np.r_[[np.real(iterative_err(T_true, V_true, x[:sys.nb], x[sys.nb:])) for x in all_x_lm],
                                        [np.nan] * (500 - len(all_x_lm))]
        all_iterative_err_lm_norm = np.r_[[np.real(iterative_err(T_true, V_true, x[:sys.nb], x[sys.nb:])) for x in all_x_lm_norm],
                                        [np.nan] * (500 - len(all_x_lm_norm))]
        all_iterative_err_gn = np.r_[[np.real(iterative_err(T_true, V_true, x[:sys.nb], x[sys.nb:])) for x in all_x_gn],
                                        [np.nan] * (500 - len(all_x_gn))]
        all_iterative_err_gn_norm = np.r_[[np.real(iterative_err(T_true, V_true, x[:sys.nb], x[sys.nb:])) for x in all_x_gn_norm],
                                        [np.nan] * (500 - len(all_x_gn_norm))]

        iterative_err_fgd.append(all_iterative_err_fgd)
        iterative_err_agd.append(all_iterative_err_agd)
        iterative_err_lm.append(all_iterative_err_lm)
        iterative_err_lm_norm.append(all_iterative_err_lm_norm)
        iterative_err_gn.append(all_iterative_err_gn)
        iterative_err_gn_norm.append(all_iterative_err_gn_norm)


        if kwargs.get('verbose', False):
            print(f'[Trial {i + 1}] Converged: {gn_converged}, RMSE GN: {rmse_gn[-1]:.4f}')
            print(f'[Trial {i + 1}] Converged: {gn_norm_converged}, RMSE GN norm: {rmse_gn_norm[-1]:.4f}, RMSE GN norm GN: {rmse_gn_norm_gn[-1]:.4f}')
            print(f'[Trial {i + 1}] Converged: {fgd_gn_converged}, RMSE FGD: {rmse_fgd[-1]:.4f}, RMSE FGD GN: {rmse_fgd_gn[-1]:.4f}')
            print(f'[Trial {i + 1}] Converged: {agd_gn_converged}, RMSE AGD: {rmse_agd[-1]:.4f}, RMSE AGD GN: {rmse_agd_gn[-1]:.4f}')
            print(f'[Trial {i + 1}] Converged: {lm_gn_converged}, RMSE LM: {rmse_lm[-1]:.4f}, RMSE LM GN: {rmse_lm_gn[-1]:.4f}')
            print(f'[Trial {i + 1}] Converged: {lm_norm_gn_converged}, RMSE LM norm: {rmse_lm_norm[-1]:.4f}, RMSE LM norm GN: {rmse_lm_norm_gn[-1]:.4f}')



    average_iterative_error_fgd = np.nanmean(np.vstack(iterative_err_fgd), axis=0)
    average_iterative_error_agd = np.nanmean(np.vstack(iterative_err_agd), axis=0)
    average_iterative_error_lm = np.nanmean(np.vstack(iterative_err_lm), axis=0)
    average_iterative_error_lm_norm = np.nanmean(np.vstack(iterative_err_lm_norm), axis=0)
    average_iterative_error_gn = np.nanmean(np.vstack(iterative_err_gn), axis=0)
    average_iterative_error_gn_norm = np.nanmean(np.vstack(iterative_err_gn_norm), axis=0)


    return (rmse_gn, rmse_gn_norm, rmse_gn_norm_gn,
            rmse_fgd, rmse_fgd_gn,
            rmse_agd, rmse_agd_gn,
            rmse_lm, rmse_lm_gn, rmse_lm_norm, rmse_lm_norm_gn,
            average_iterative_error_gn, average_iterative_error_gn_norm,
            average_iterative_error_fgd, average_iterative_error_agd,
            average_iterative_error_lm, average_iterative_error_lm_norm)


def main():
    data = parse_ieee_mat(file)
    system_data = data['data']['system']
    sys = System(system_data)
    branch = Branch(sys.branch)
    init_params= (.35, .95, 1.05)
    kwargs = {'flow': True, 'injection': False, 'voltage': True,
              'noise': True, 'Pf_noise': 4e-4, 'Qf_noise': 4e-4, 'Cm_noise': 1e-4,
              'Pi_noise': 16e-4, 'Qi_noise': 16e-4, 'Vm_noise': 1.6e-05, 'verbose': True}
    (rmse_gn, rmse_gn_norm, rmse_gn_norm_gn,
     rmse_fgd, rmse_fgd_gn,
     rmse_agd, rmse_agd_gn,
     rmse_lm, rmse_lm_gn, rmse_lm_norm, rmse_lm_norm_gn,
     average_iterative_error_gn, average_iterative_error_gn_norm,
     average_iterative_error_fgd, average_iterative_error_agd,
     average_iterative_error_lm, average_iterative_error_lm_norm) = \
        run_experiment(data, sys, branch, init_params, **kwargs)

    mean_rmse_gn = np.mean(rmse_gn)
    mean_rmse_gn_norm, mean_rmse_gn_norm_gn = np.mean(rmse_gn_norm), np.mean(rmse_gn_norm_gn)
    mean_rmse_fgd, mean_rmse_fgd_gn = np.mean(rmse_fgd), np.mean(rmse_fgd_gn)
    mean_rmse_agd, mean_rmse_agd_gn = np.mean(rmse_agd), np.mean(rmse_agd_gn)
    mean_rmse_lm, mean_rmse_lm_gn = np.mean(rmse_lm), np.mean(rmse_lm_gn)
    mean_rmse_lm_norm, mean_rmse_lm_norm_gn = np.mean(rmse_lm_norm), np.mean(rmse_lm_norm_gn)


    print(f'Mean RMSE GN: {mean_rmse_gn:.4f}')
    print(f'Mean RMSE GN-norm: {mean_rmse_gn_norm:.4f}, Mean RMSE GN-norm-GN: {mean_rmse_gn_norm_gn:.4f}')
    print(f'Mean RMSE FGD: {mean_rmse_fgd:.4f}, Mean RMSE FGD-GN: {mean_rmse_fgd_gn:.4f}')
    print(f'Mean RMSE AGD: {mean_rmse_agd:.4f}, Mean RMSE AGD-GN: {mean_rmse_agd_gn:.4f}')
    print(f'Mean RMSE LM: {mean_rmse_lm:.4f}, Mean RMSE LM-GN: {mean_rmse_lm_gn:.4f}')
    print(f'Mean RMSE LM-norm: {mean_rmse_lm_norm:.4f}, Mean RMSE LM-norm-GN: {mean_rmse_lm_norm_gn:.4f}')
    # Create figure with two subplots: main plot + histogram
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])

    # --- Convergence Plot ---
    ax0 = plt.subplot(gs[0])

    ax0.plot(average_iterative_error_gn, label='GN', linestyle='-')
    ax0.plot(average_iterative_error_gn_norm, label='GN-norm', linestyle='--')
    ax0.plot(average_iterative_error_fgd, label='FGD', linestyle='-.')
    ax0.plot(average_iterative_error_agd, label='AGD', linestyle=':')
    ax0.plot(average_iterative_error_lm, label='LM', linestyle=(0, (3, 1, 1, 1)))  # dash-dot-dot
    ax0.plot(average_iterative_error_lm_norm, label='LM-norm', linestyle=(0, (5, 2)))  # custom dashed

    ax0.set_xlabel('Number of Iterations')
    ax0.set_ylabel(r'$\frac{\|V_1 - V\|_F}{\|V\|_F}$')
    x_max = max(len(average_iterative_error_fgd), len(average_iterative_error_agd))
    ax0.set_xticks(np.arange(0, x_max + 1, 100))
    y_max = max(np.nanmax(average_iterative_error_fgd), np.nanmax(average_iterative_error_agd))
    ax0.set_yticks(np.arange(0, y_max + 0.1, 0.1))
    ax0.grid(True, ls=':', alpha=0.7)
    ax0.legend()
    ax0.set_title("Convergence of Iterative Methods")

    # --- RMSE Histogram ---
    ax1 = plt.subplot(gs[1])

    rmse_data = {
        'GN': rmse_gn,
        'GN-norm': rmse_gn_norm,
        'FGD': rmse_fgd,
        'AGD': rmse_agd,
        'LM': rmse_lm,
        'LM-norm': rmse_lm_norm,
    }

    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd', '#8c564b']
    ax1.hist(rmse_data.values(), bins=20, label=rmse_data.keys(), color=colors, stacked=True, orientation='horizontal',
             alpha=0.8)
    ax1.set_xlabel('Frequency')
    ax1.set_yticks([])
    ax1.set_title('RMSE Distribution')
    ax1.legend(fontsize='small')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()






