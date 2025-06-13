import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from power_flow_ac.process_net_data import parse_ieee_mat, System, Branch
from power_flow_ac.power_flow_cartesian import H_AC as H_AC_cartesian
from power_flow_ac.power_flow_polar import H_AC as H_AC_polar
from SDP_se import SDP_se
from GN_se import GN_se
from power_flow_ac.init_starting_point import init_start_point

file = "C:/Users/shirsc/PycharmProjects/PSSE_proj/nets/ieee30_41.mat"

def generate_data(data, sys, branch, init_params, **kwargs):
    T_true, V_true = init_start_point(sys, data, how='random', random_init=init_params)
    Vc_true = V_true * np.exp(1j * T_true)
    meas_idx = {}
    if kwargs.get('flow'):
        meas_idx['Pf_idx'] = np.ones(len(branch.i)).astype(bool)
        meas_idx['Qf_idx'] = np.ones(len(branch.i)).astype(bool)
    if kwargs.get('injection'):
        meas_idx['Pi_idx'] = np.ones(len(sys.bus)).astype(bool)
        meas_idx['Qi_idx'] = np.ones(len(sys.bus)).astype(bool)
    if kwargs.get('voltage'):
        meas_idx['Vm_idx'] = np.ones(len(sys.bus)).astype(bool)
    if kwargs.get('current'):
        meas_idx['Cm_idx'] = np.ones(len(branch.i)).astype(bool)

    agg_meas_idx = {}
    last_idx = 0
    meas_types = ['Pf', 'Qf', 'Cm', 'Pi', 'Qi', 'Vm']
    for k in meas_types:
        v = meas_idx.get(f'{k}_idx', [])
        agg_meas_idx[k] = np.arange(last_idx, last_idx + len(v))
        last_idx += len(v)

    h_ac_cart = H_AC_cartesian(sys, branch, meas_idx)
    h_ac_polar = H_AC_polar(sys, branch, meas_idx)
    z, _ = h_ac_polar.estimate(V=V_true, T=T_true)
    var = np.zeros(len(z))
    if kwargs.get('noise'):
        var = [
            np.repeat(kwargs.get(f'{meas_type}_noise', 1),
                      len(agg_meas_idx[meas_type]))
            for meas_type in meas_types
        ]
        var = np.concatenate(var)
        noise = np.random.normal(np.zeros_like(z), np.sqrt(var))
        z += noise

    return z, var, meas_idx, agg_meas_idx, h_ac_cart, h_ac_polar, T_true, V_true, Vc_true


def square_mag(z, var, agg_meas_idx):
    z_square = z.copy()
    mag_idx = np.r_[agg_meas_idx['Cm'], agg_meas_idx['Vm']]
    z_square[mag_idx] = z[mag_idx] ** 2
    var_square = var.copy()
    var_square[mag_idx] = 2 * (var[mag_idx] ** 2)
    return z_square, var_square


def calc_dT(T_true, T_est):
    T_true_deg = np.rad2deg(T_true)
    T_est_deg = np.rad2deg(T_est)
    return np.deg2rad((T_true_deg - T_est_deg + 180) % 360 - 180)
    # return T_true - T_est

def RMSE(T_true, V_true, T_est, V_est, nb):
    dT = calc_dT(T_true, T_est)
    dV = V_true - V_est

    return np.sqrt(np.sum((np.r_[dT, dV]) ** 2) / (2 * nb))


def sample_from_SDR(Vc_est, V_opt, sys, T_true, V_true, num_samples=50):
    V_real, V_imag = np.real(Vc_est), np.imag(Vc_est)

    T = np.arctan2(V_imag, V_real)
    V = np.sqrt(V_real ** 2 + V_imag ** 2)

    best_fit = RMSE(T_true, V_true, T, V, sys.nb)

    T_best = T
    V_best = V

    for _ in range(num_samples):
        mu = np.zeros(sys.nb * 2)
        cov = .5 * np.block([[np.real(V_opt), -np.imag(V_opt)],
                             [np.imag(V_opt), np.real(V_opt)]])
        Vc = np.random.multivariate_normal(mu, cov)
        Vc = Vc[:sys.nb] + 1j * Vc[sys.nb:]
        V_real, V_imag = np.real(Vc), np.imag(Vc)
        T = np.arctan2(V_imag, V_real)
        V = np.sqrt(V_real ** 2 + V_imag ** 2)

        fit = RMSE(T_true, V_true, T, V, sys.nb)

        if fit < best_fit:
            best_fit = fit
            T_best = T
            V_best = V

    return T_best, V_best

def run_experiment(data, sys, branch, init_params, **kwargs):
    rmse_sdr = []
    rmse_sdr_wls = []
    rmse_flt_wls = []

    angle_error_sdr = []
    angle_error_sdr_wls = []
    angle_error_flt_wls = []

    mag_error_sdr = []
    mag_error_sdr_wls = []
    mag_error_flt_wls = []

    converged_sdr = []
    converged_sdr_wls = []
    converged_flt_wls = []

    for i in tqdm(range(100)):
        sdr_converged = False
        while not sdr_converged:
            gen_data = generate_data(data, sys, branch, init_params, **kwargs)
            z_wls, var_wls, meas_idx, agg_meas_idx, h_ac_cart, h_ac_polar, T_true, V_true, Vc_true = gen_data
            z_sdr, var_sdr = square_mag(z_wls, var_wls, agg_meas_idx)

            V_est, V_opt, sdr_converged = SDP_se(z_sdr, var_sdr, sys.slk_bus, h_ac_cart, sys.nb)
        T_sdr, V_sdr = sample_from_SDR(V_est, V_opt, sys, T_true, V_true)

        x_sdr = np.r_[T_sdr, V_sdr]
        T_sdr_wls, V_sdr_wls, _, _, wls_sdr_converged = GN_se(z_wls, var_wls, sys.slk_bus, h_ac_polar, sys.nb, x0=x_sdr)

        x_flat = np.r_[init_start_point(sys, data, how='flat')]
        T_flat, V_flat, _, _, wls_converged = GN_se(z_wls, var_wls, sys.slk_bus, h_ac_polar, sys.nb, x0=x_flat)


        converged_sdr.append(sdr_converged)
        converged_sdr_wls.append(wls_sdr_converged)
        converged_flt_wls.append(wls_converged)

        rmse_sdr.append(RMSE(T_true, V_true, T_sdr, V_sdr, sys.nb))
        rmse_sdr_wls.append(RMSE(T_true, V_true, T_sdr_wls, V_sdr_wls, sys.nb))
        rmse_flt_wls.append(RMSE(T_true, V_true, T_flat, V_flat, sys.nb))

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

    mean_rmse_sdr = np.mean(rmse_sdr)
    mean_rmse_sdr_wls = np.mean(rmse_sdr_wls)
    mean_rmse_flt_wls = np.mean(rmse_flt_wls)

    mean_angle_error_sdr = np.mean(np.vstack(angle_error_sdr), axis=0)
    mean_angle_error_sdr_wls = np.mean(np.vstack(angle_error_sdr_wls), axis=0)
    mean_angle_error_flt_wls = np.mean(np.vstack(angle_error_flt_wls), axis=0)
    mean_mag_error_sdr = np.mean(np.vstack(mag_error_sdr), axis=0)
    mean_mag_error_sdr_wls = np.mean(np.vstack(mag_error_sdr_wls), axis=0)
    mean_mag_error_flt_wls = np.mean(np.vstack(mag_error_flt_wls), axis=0)

    mean_converged_sdr = np.mean(np.array(converged_sdr).astype(int))
    mean_converged_sdr_wls = np.mean(np.array(converged_sdr_wls).astype(int))
    mean_converged_flt_wls = np.mean(np.array(converged_flt_wls).astype(int))

    return mean_rmse_sdr, mean_rmse_sdr_wls, mean_rmse_flt_wls, \
           mean_angle_error_sdr, mean_angle_error_sdr_wls, mean_angle_error_flt_wls, \
           mean_mag_error_sdr, mean_mag_error_sdr_wls, mean_mag_error_flt_wls, \
           mean_converged_sdr, mean_converged_sdr_wls, mean_converged_flt_wls


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
        mean_rmse_sdr, mean_rmse_sdr_wls, mean_rmse_flt_wls, \
        mean_angle_error_sdr, mean_angle_error_sdr_wls, mean_angle_error_flt_wls, \
        mean_mag_error_sdr, mean_mag_error_sdr_wls, mean_mag_error_flt_wls,\
        mean_converged_sdr, mean_converged_sdr_wls, mean_converged_flt_wls = \
            run_experiment(data, sys, branch, init_params, **kwargs)
        print(f'Converged {mean_converged_sdr * 100}%, Mean RMSE SDR: {mean_rmse_sdr:.4e}')
        print(f'Converged {mean_converged_sdr_wls * 100}%, Mean RMSE WLS SDR: {mean_rmse_sdr_wls:.4e}')
        print(f'Converged {mean_converged_flt_wls * 100}%, Mean RMSE WLS FLT: {mean_rmse_flt_wls:.4e}')


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






