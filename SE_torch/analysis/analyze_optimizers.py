import torch

from SE_torch.PF_equations.PF_cartesian import H_AC as H_AC_cartesian
from SE_torch.data_generator import DataGenerator
from SE_torch.net_preprocess.process_net_data import parse_ieee_mat, System, Branch
from SE_torch.utils import square_mag, normalize_measurements, RMSE, init_start_point
from SE_torch.optimizers.GN_se import GN_se
from SE_torch.optimizers.FO_se import SGD_se


def run_experiment(data, sys, branch, data_generator, **kwargs):
    T0, V0 = init_start_point(sys, data, how='flat')

    gen_data = data_generator.generate_measurements(sys, branch, device=None, **kwargs)
    z, v, meas_idx, agg_meas_idx, h_ac_cart, h_ac_polar, T_true, V_true, Vc_true = gen_data
    measurements_square, variance_square = square_mag(z, v, agg_meas_idx)
    h_ac_cart_norm = H_AC_cartesian(sys, branch, meas_idx)
    measurements_square_norm, h_ac_cart_norm.H, norm_H = normalize_measurements(h_ac_cart.H, measurements_square)

    x0 = torch.concatenate([T0, V0])

    u0 = V0 * torch.exp(1j * T0)
    x_gn, T_gn, V_gn, gn_converged, k_gn = GN_se(x0, z, v, sys.slk_bus, h_ac_polar, sys.nb)
    rmse_gn = RMSE(T_true, V_true, T_gn, V_gn)

    x_sgd, T_sgd, V_sgd, sgd_converged, k_sgd = SGD_se(x0, z, v, sys.slk_bus, h_ac_polar, sys.nb)
    rmse_sgd = RMSE(T_true, V_true, T_sgd, V_sgd)

    return rmse_gn, rmse_sgd, k_gn, k_sgd, gn_converged, sgd_converged


def main(net_file, num_experiments, verbose=False):
    data = parse_ieee_mat(net_file)
    system_data = data['data']['system']
    sys = System(system_data)
    branch = Branch(sys.branch)
    data_generator = DataGenerator()
    kwargs = {'flow': True, 'injection': True, 'voltage': True, 'current': False,
              'noise': True, 'Pf_noise': 4e-4, 'Qf_noise': 4e-4, 'Cm_noise': 1e-4,
              'Pi_noise': 16e-4, 'Qi_noise': 16e-4, 'Vm_noise': 1.6e-05, 'verbose': True}

    all_rmse_gn, all_rmse_sgd = [], []
    all_k_gn, all_k_sgd = [], []

    for i in range(num_experiments):
        rmse_gn, rmse_sgd, k_gn, k_sgd, gn_conv, sgd_conv = run_experiment(data, sys, branch, data_generator, **kwargs)
        all_rmse_gn.append(rmse_gn)
        all_rmse_sgd.append(rmse_sgd)
        all_k_gn.append(k_gn)
        all_k_sgd.append(k_sgd)

        if verbose:
            print(f'[Trial {i}] Converged: {gn_conv}, RMSE GN: {rmse_gn:.8f}, steps: {k_gn}')
            print(f'[Trial {i}] Converged: {sgd_conv}, RMSE SGD: {rmse_sgd:.8f}, steps: {k_sgd}')

    mean_rmse_gn = torch.nanmean(torch.Tensor(all_rmse_gn))
    mean_rmse_sgd = torch.nanmean(torch.Tensor(all_rmse_sgd))
    mean_k_gn = torch.nanmean(torch.Tensor(all_k_gn))
    mean_k_sgd = torch.nanmean(torch.Tensor(all_k_sgd))

    if verbose:
        print(f'Mean RMSE GN: {mean_rmse_gn:.8f}, Mean steps: {mean_k_gn}')
        print(f'Mean RMSE SGD: {mean_rmse_sgd:.8f}, Mean steps: {mean_k_sgd}')

if __name__ == '__main__':
    file = "../../nets/ieee118_186.mat"
    num_exp = 1
    verbose = True
    main(file, num_exp, verbose)
