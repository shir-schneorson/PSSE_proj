import json

import torch

from SE_torch.PF_equations.PF_cartesian import H_AC as H_AC_cartesian
from SE_torch.data_generator import DataGenerator
from SE_torch.net_preprocess.process_net_data import parse_ieee_mat, System, Branch
from SE_torch.utils import square_mag, normalize_measurements, RMSE, init_start_point
from SE_torch.optimizers.GN_se import GN_se
from sandbox.GN_cart_se import GN_cart_se


def run_experiment(data, sys, branch, data_generator, optimizers, **kwargs):
    T0, V0 = init_start_point(sys, data, how='flat')
    T0, V0 = T0.to(torch.float64), V0.to(torch.float64)

    gen_data = data_generator.generate_measurements(sys, branch, device=None, **kwargs)
    z, z_cart, v, meas_idx, agg_meas_idx, h_ac_cart, h_ac_polar, T_true, V_true, Vc_true = gen_data
    measurements_square, variance_square = square_mag(z, v, agg_meas_idx)
    h_ac_cart_norm = H_AC_cartesian(sys, branch, meas_idx)
    measurements_square_norm, h_ac_cart_norm.H, norm_H = normalize_measurements(h_ac_cart.H, measurements_square)

    x0 = torch.concatenate([T0, V0])
    # u0 = V0 * torch.exp(1j * T0)
    rmse, k, converged, conds = {}, {}, {}, {}

    x_polar, T_polar, V_polar, polar_converged, k_polar, polar_conds = optimizers['gn_polar'](x0, z, v, sys.slk_bus, h_ac_polar, sys.nb)
    rmse['gn_polar'] = RMSE(T_true, V_true, T_polar.detach(), V_polar.detach())
    k['gn_polar'] = k_polar # + k_gn
    converged['gn_polar'] = polar_converged # * gn_converged
    conds['gn_polar'] = polar_conds

    x_cart, T_cart, V_cart, cart_converged, k_cart, carts_conds = optimizers['gn_cart'](x0, z_cart, v, sys.slk_bus, h_ac_cart, sys.nb)
    rmse['gn_cart'] = RMSE(T_true, V_true, T_cart.detach(), V_cart.detach())
    k['gn_cart'] = k_cart # + k_gn
    converged['gn_cart'] = cart_converged # * gn_converged
    conds['gn_cart'] = carts_conds

    return rmse, k, converged, conds


def run_all_experiments(config):
    net_file = config['net_file']
    num_experiments = config['num_experiments']
    verbose = config['verbose']
    data = parse_ieee_mat(net_file)
    system_data = data['data']['system']
    sys = System(system_data)
    branch = Branch(sys.branch)
    data_generator = DataGenerator()
    kwargs = config["generator_kwargs"]

    optimizers = {
        'gn_polar': GN_se(verbose=False, max_iter=100),
        'gn_cart': GN_cart_se(verbose=False, max_iter=100),
    }
    all_rmse = {opt: [] for opt in optimizers.keys()}
    all_k = {opt: [] for opt in optimizers.keys()}
    all_conv = {opt: [] for opt in optimizers.keys()}
    all_first_conds = {opt: [] for opt in optimizers.keys()}
    all_last_conds = {opt: [] for opt in optimizers.keys()}

    for i in range(num_experiments):
        rmse, k, conv, conds = run_experiment(data, sys, branch, data_generator, optimizers, **kwargs)
        print()
        for opt in optimizers.keys():
            all_rmse[opt].append(rmse[opt])
            all_k[opt].append(k[opt])
            all_conv[opt].append(conv[opt])
            all_first_conds[opt].append(conds[opt][0])
            all_last_conds[opt].append(conds[opt][-1])

            if verbose:
                print(f'[Trial {i}] [{opt.upper()}] Converged: {bool(conv[opt])}, RMSE: {rmse[opt]:.8f}, steps: {k[opt]}, first-last cond: {conds[opt][0]:.3e}, {conds[opt][-1]:.3e}')
    print()
    for opt in optimizers.keys():
        all_rmse_opt = torch.Tensor(all_rmse[opt])
        all_k_opt = torch.Tensor(all_k[opt])
        all_conv_opt = torch.Tensor(all_conv[opt])
        all_first_cond_opt = torch.Tensor(all_first_conds[opt])
        all_last_cond_opt = torch.Tensor(all_last_conds[opt])
        # all_rmse_opt = all_rmse_opt[all_rmse_opt < torch.quantile(all_rmse_opt, 0.95)]
        mean_rmse = torch.nanmean(all_rmse_opt)
        mean_k = torch.nanmean(all_k_opt)
        mean_conv = torch.nanmean(all_conv_opt)
        mean_first_cond = torch.nanmean(all_first_cond_opt)
        mean_last_cond = torch.nanmean(all_last_cond_opt)

        if verbose:
            print(f'[{opt.upper()}]  Mean RMSE: {mean_rmse:.8f}, Mean steps: {mean_k:.2f}, Mean convergence: {mean_conv:.2f}, mean first-last cond: {mean_first_cond:.3e}, {mean_last_cond:.3e}')

def main():
    config = json.load(open("./configs/analyze_optimizer_cart_vs_polar_config.json"))
    run_all_experiments(config)

if __name__ == '__main__':
    main()
