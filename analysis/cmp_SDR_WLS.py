import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from optimizers.SGD_se import FGD_se
from init_net.process_net_data import parse_ieee_mat, System, Branch
from init_net.power_flow_cartesian import H_AC as H_AC_cartesian
from optimizers.SDP_se import SDP_se
from optimizers.GN_se import GN_se
from init_net.init_starting_point import init_start_point
from utils import generate_data, square_mag, sample_from_SGD, RMSE, normalize_measurements, calc_dT, sample_from_SDR


file = "/Users/shirschneorson/PycharmProjects/PSSE_proj/nets/ieee118_186.mat"


def run_experiment(data, sys, branch, init_params, **kwargs):
    rmse_sdr = []
    rmse_sdr_wls = []
    rmse_flt_wls = []
    rmse_fgd_wls = []
    rmse_agd_wls = []

    angle_error_sdr = []
    angle_error_sdr_wls = []
    angle_error_flt_wls = []

    mag_error_sdr = []
    mag_error_sdr_wls = []
    mag_error_flt_wls = []


    converged_sdr = []
    converged_sdr_wls = []
    converged_flt_wls = []
    converged_fgd_wls = []
    converged_agd_wls = []

    for i in tqdm(range(100)):
        sdr_converged = False
        while not sdr_converged:
            gen_data = generate_data(data, sys, branch, init_params, **kwargs)
            z_wls, var_wls, meas_idx, agg_meas_idx, h_ac_cart, h_ac_polar, T_true, V_true, Vc_true = gen_data
            z_sdr, var_sdr = square_mag(z_wls, var_wls, agg_meas_idx)

            V_est, V_opt, sdr_converged = SDP_se(z_sdr, var_sdr, sys.slk_bus, h_ac_cart, sys.nb)
        T_sdr, V_sdr = sample_from_SDR(V_est, V_opt, sys, T_true, V_true)

        h_ac_cart_gd = H_AC_cartesian(sys, branch, meas_idx)
        z_gd, h_ac_cart_gd.H = normalize_measurements(z_sdr, h_ac_cart_gd.H)

        T0, V0 = init_start_point(sys, data, how='flat')
        u0 = V0 * np.exp(1j * T0)

        u_fgd, converged_fgd = FGD_se(u0, z_gd, h_ac_cart_gd)
        u_agd, converged_age = FGD_se(u0, z_gd, h_ac_cart_gd, AGD_update=True)

        u_fgd, T_fgd, V_fgd, err_fgd = sample_from_SGD(u_fgd, h_ac_cart_gd.H, z_gd)
        u_agd, T_agd, V_agd, err_agd = sample_from_SGD(u_agd, h_ac_cart_gd.H, z_gd)

        x_fgd = np.r_[T_fgd, V_fgd]
        T_fgd_wls, V_fgd_wls, _, _, wls_fgd_converged = GN_se(z_wls, var_wls, sys.slk_bus, h_ac_polar, sys.nb,
                                                              x0=x_fgd)
        x_agd = np.r_[T_agd, V_agd]
        T_agd_wls, V_agd_wls, _, _, wls_agd_converged = GN_se(z_wls, var_wls, sys.slk_bus, h_ac_polar, sys.nb,
                                                              x0=x_agd)
        x_sdr = np.r_[T_sdr, V_sdr]
        T_sdr_wls, V_sdr_wls, _, _, wls_sdr_converged = GN_se(z_wls, var_wls, sys.slk_bus, h_ac_polar, sys.nb, x0=x_sdr)

        x_flat = np.r_[T0, V0]
        T_flat, V_flat, _, _, wls_converged = GN_se(z_wls, var_wls, sys.slk_bus, h_ac_polar, sys.nb, x0=x_flat)

        converged_sdr.append(sdr_converged)
        converged_sdr_wls.append(wls_sdr_converged)
        converged_flt_wls.append(wls_converged)
        converged_fgd_wls.append(wls_fgd_converged)
        converged_agd_wls.append(wls_agd_converged)

        rmse_sdr.append(RMSE(T_true, V_true, T_sdr, V_sdr, sys.nb))
        rmse_sdr_wls.append(RMSE(T_true, V_true, T_sdr_wls, V_sdr_wls, sys.nb))
        rmse_flt_wls.append(RMSE(T_true, V_true, T_flat, V_flat, sys.nb))
        rmse_fgd_wls.append(RMSE(T_true, V_true, T_fgd_wls, V_fgd_wls, sys.nb))
        rmse_agd_wls.append(RMSE(T_true, V_true, T_agd_wls, V_agd_wls, sys.nb))

        angle_error_sdr.append(np.sqrt(np.rad2deg(calc_dT(T_true, T_sdr)) ** 2))
        angle_error_sdr_wls.append(np.sqrt(np.rad2deg(calc_dT(T_true, T_sdr_wls)) ** 2))
        angle_error_flt_wls.append(np.sqrt(np.rad2deg(calc_dT(T_true, T_flat)) ** 2))
        mag_error_sdr.append(np.sqrt((V_true - V_sdr) ** 2))
        mag_error_sdr_wls.append(np.sqrt((V_true - V_sdr_wls) ** 2))
        mag_error_flt_wls.append(np.sqrt((V_true - V_flat) ** 2))

        if kwargs.get('verbose', False):
            print(f'[Trial {i + 1}] Converged: {converged_sdr[-1]}, RMSE SDR: {rmse_sdr[-1]:.4e}')
            print(f'[Trial {i + 1}] Converged: {converged_sdr_wls[-1]}, RMSE WLS SDR: {rmse_sdr_wls[-1]:.4e}')
            print(f'[Trial {i + 1}] Converged: {converged_flt_wls[-1]}, RMSE WLS FLT: {rmse_flt_wls[-1]:.4e}')
            print(f'[Trial {i + 1}] Converged: {converged_fgd_wls[-1]}, RMSE WLS FGD: {rmse_fgd_wls[-1]:.4e}')
            print(f'[Trial {i + 1}] Converged: {converged_agd_wls[-1]}, RMSE WLS FGD: {rmse_agd_wls[-1]:.4e}')

    mean_rmse_sdr = np.mean(rmse_sdr)
    mean_rmse_sdr_wls = np.mean(rmse_sdr_wls)
    mean_rmse_flt_wls = np.mean(rmse_flt_wls)
    mean_rmse_fgd_wls = np.mean(rmse_fgd_wls)
    mean_rmse_agd_wls = np.mean(rmse_agd_wls)

    mean_angle_error_sdr = np.mean(np.vstack(angle_error_sdr), axis=0)
    mean_angle_error_sdr_wls = np.mean(np.vstack(angle_error_sdr_wls), axis=0)
    mean_angle_error_flt_wls = np.mean(np.vstack(angle_error_flt_wls), axis=0)

    mean_mag_error_sdr = np.mean(np.vstack(mag_error_sdr), axis=0)
    mean_mag_error_sdr_wls = np.mean(np.vstack(mag_error_sdr_wls), axis=0)
    mean_mag_error_flt_wls = np.mean(np.vstack(mag_error_flt_wls), axis=0)

    mean_converged_sdr = np.mean(np.array(converged_sdr).astype(int))
    mean_converged_sdr_wls = np.mean(np.array(converged_sdr_wls).astype(int))
    mean_converged_flt_wls = np.mean(np.array(converged_flt_wls).astype(int))
    mean_converged_fgd_wls = np.mean(np.array(converged_fgd_wls).astype(int))
    mean_converged_agd_wls = np.mean(np.array(converged_agd_wls).astype(int))

    return (mean_rmse_sdr, mean_rmse_sdr_wls, mean_rmse_flt_wls, mean_rmse_fgd_wls, mean_rmse_agd_wls,
           mean_angle_error_sdr, mean_angle_error_sdr_wls, mean_angle_error_flt_wls,
           mean_mag_error_sdr, mean_mag_error_sdr_wls, mean_mag_error_flt_wls,
           mean_converged_sdr, mean_converged_sdr_wls, mean_converged_flt_wls, mean_converged_fgd_wls, mean_converged_agd_wls)


def main():
    data = parse_ieee_mat(file)
    system_data = data['data']['system']
    sys = System(system_data)
    branch = Branch(sys.branch)
    init_params_all = [(.5, 1, 1e-2), (.4, 1, 1e-2), (.3, 1, 1e-2)]
    kwargs = {'flow': True, 'injection': False, 'voltage': True,
              'noise': True, 'Pf_noise': 4e-4, 'Qf_noise': 4e-4, 'Cm_noise': 1e-4,
              'Pi_noise': 4e-4, 'Qi_noise': 4e-4, 'Vm_noise': 1e-4, 'verbose': True}
    for init_params in init_params_all:
        (mean_rmse_sdr, mean_rmse_sdr_wls, mean_rmse_flt_wls, mean_rmse_fgd_wls, mean_rmse_agd_wls,
         mean_angle_error_sdr, mean_angle_error_sdr_wls, mean_angle_error_flt_wls,
         mean_mag_error_sdr, mean_mag_error_sdr_wls, mean_mag_error_flt_wls,
         mean_converged_sdr, mean_converged_sdr_wls, mean_converged_flt_wls, mean_converged_fgd_wls,
         mean_converged_agd_wls) = \
            run_experiment(data, sys, branch, init_params, **kwargs)
        print(f'Converged {mean_converged_sdr * 100}%, Mean RMSE SDR: {mean_rmse_sdr:.4e}')
        print(f'Converged {mean_converged_sdr_wls * 100}%, Mean RMSE WLS SDR: {mean_rmse_sdr_wls:.4e}')
        print(f'Converged {mean_converged_flt_wls * 100}%, Mean RMSE WLS FLT: {mean_rmse_flt_wls:.4e}')
        print(f'Converged {mean_converged_fgd_wls * 100}%, Mean RMSE WLS FGD: {mean_rmse_fgd_wls:.4e}')
        print(f'Converged {mean_converged_agd_wls * 100}%, Mean RMSE WLS AGD: {mean_rmse_agd_wls:.4e}')


        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(rf"Angle and Magnitude Error {init_params[0]}\$pi\$")
        buses = np.arange(len(sys.bus))
        axes[0].plot(buses, mean_angle_error_sdr, 'o-', label='SDR')
        axes[0].plot(buses, mean_angle_error_sdr_wls, 'v-',  label='WLS SDR')
        axes[0].plot(buses, mean_angle_error_flt_wls, '*-', label='WLS FLT')
        axes[0].set_xlabel('Bus')
        axes[0].set_ylabel('Angle Error (deg)')
        axes[0].legend()

        axes[1].plot(buses, mean_mag_error_sdr, 'o-', label='SDR')
        axes[1].plot(buses, mean_mag_error_sdr_wls, 'v-',  label='WLS SDR')
        axes[1].plot(buses, mean_mag_error_flt_wls, '*-', label='WLS FLT')
        axes[1].set_xlabel('Bus')
        axes[1].set_ylabel('Magnitude Error (pu)')
        axes[1].legend()
        plt.show()


if __name__ == '__main__':
    main()






