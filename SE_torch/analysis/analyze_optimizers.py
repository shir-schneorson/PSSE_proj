import torch

from SE_torch.PF_equations.PF_cartesian import H_AC as H_AC_cartesian
from SE_torch.data_generator import DataGenerator
from SE_torch.net_preprocess.process_net_data import parse_ieee_mat, System, Branch
from SE_torch.utils import square_mag, normalize_measurements, RMSE, init_start_point
from SE_torch.optimizers.GN_se import GN_se
from SE_torch.optimizers.LM_se import LM_se
from SE_torch.optimizers.FO_se import SGD_se, AdamW_se, Muon_se, LBFGS_se


def run_experiment(data, sys, branch, data_generator, optimizers, **kwargs):
    T0, V0 = init_start_point(sys, data, how='flat')
    # T0, V0 = data_generator.sample(sys, num_samples=1, random_flow=True)
    T0 = T0.to(torch.float64)
    V0 = V0.to(torch.float64)

    gen_data = data_generator.generate_measurements(sys, branch, device=None, **kwargs)
    z, v, meas_idx, agg_meas_idx, h_ac_cart, h_ac_polar, T_true, V_true, Vc_true = gen_data
    measurements_square, variance_square = square_mag(z, v, agg_meas_idx)
    h_ac_cart_norm = H_AC_cartesian(sys, branch, meas_idx)
    measurements_square_norm, h_ac_cart_norm.H, norm_H = normalize_measurements(h_ac_cart.H, measurements_square)

    x0 = torch.concatenate([T0, V0])

    u0 = V0 * torch.exp(1j * T0)
    rmse, k, converged = {}, {}, {}
    for opt_name, optimizer in optimizers.items():
        x_opt, T_opt, V_opt, opt_converged, k_opt = optimizer(x0, z, v, sys.slk_bus, h_ac_polar, sys.nb)
        # x_opt_gn, T_opt, V_opt, gn_converged, k_gn = optimizers['gn'](x_opt, z, v, sys.slk_bus, h_ac_polar, sys.nb)
        rmse[opt_name] = RMSE(T_true, V_true, T_opt.detach(), V_opt.detach())
        k[opt_name] = k_opt # + k_gn
        converged[opt_name] = opt_converged # * gn_converged

    return rmse, k, converged


def main(net_file, num_experiments, verbose=False):
    data = parse_ieee_mat(net_file)
    system_data = data['data']['system']
    sys = System(system_data)
    branch = Branch(sys.branch)
    data_generator = DataGenerator()
    kwargs = {'flow': True, 'injection': True, 'voltage': True, 'current': False,
              'noise': True, 'Pf_noise': 4e-3, 'Qf_noise': 4e-3, 'Cm_noise': 1e-3,
              'Pi_noise': 16e-3, 'Qi_noise': 16e-3, 'Vm_noise': 1.6e-03, 'verbose': True, 'sample': 0.7}
    m, cov = torch.load('../learn_prior/datasets/mean.pt'), torch.load(
        '../learn_prior/datasets/cov.pt')
    # kwargs = {'flow': True, 'injection': True, 'voltage': True, 'current': False,
    #           'noise': True, 'Pf_noise': 4e-4, 'Qf_noise': 4e-4, 'Cm_noise': 1e-4,
    #           'Pi_noise': 16e-4, 'Qi_noise': 16e-4, 'Vm_noise': 1.6e-05, 'verbose': True, 'sample': 0.7}
    # kwargs = {'flow': True, 'injection': True, 'voltage': True, 'current': False,
    #           'noise': True, 'Pf_noise': 1e-6, 'Qf_noise': 1e-6, 'Cm_noise': 1e-6,
    #           'Pi_noise': 1e-6, 'Qi_noise': 1e-6, 'Vm_noise': 1e-6, 'verbose': True, 'sample': .8}
    # optimizers = {'gn': GN_se(verbose=False, max_iter=100),
    #               'lm_gs': LM_se(verbose=False, use_prior=True, m=m, Q=cov),
    #               'lbfgs_nf_xs': LBFGS_se(verbose=False, use_prior=True,
    #                                       prior_config_path="../learn_prior/configs/NF_2_0_8.json"),
    #               'lbfgs_nf_s': LBFGS_se(verbose=False, use_prior=True,
    #                                       prior_config_path="../learn_prior/configs/NF_2_0_64.json"),
    #               'lbfgs_nf_m': LBFGS_se(verbose=False, use_prior=True,
    #                                       prior_config_path="../learn_prior/configs/NF_4_0_64.json")
    #               }
    # optimizers = {f'lbfgs_nf_{i}_{j}_{k}':
    #                   LBFGS_se(verbose=False, use_prior=True,
    #                            prior_config_path=f"../learn_prior/configs/NF_{i}_{j}_{k}.json")
    #               for i in [2, 4, 8] for j in [0, 2, 4] for k in [8, 64, 128]
    #               }
    optimizers = {}
    # optimizers['lm_gs'] = LM_se(verbose=False, use_prior=True, m=m, Q=cov)
    optimizers['gn'] = GN_se(verbose=False, max_iter=100)
                  # 'adamw': AdamW_se(verbose=True, use_prior=True),
                  # 'sgd': SGD_se(verbose=True, use_prior=True),}
                  # 'muon': Muon_se(verbose=False),}
    all_rmse = {opt: [] for opt in optimizers.keys()}
    all_k = {opt: [] for opt in optimizers.keys()}
    all_conv = {opt: [] for opt in optimizers.keys()}

    for i in range(num_experiments):
        rmse, k, conv = run_experiment(data, sys, branch, data_generator, optimizers, **kwargs)
        for opt in optimizers.keys():
            # if conv[opt]:
            all_rmse[opt].append(rmse[opt])
            all_k[opt].append(k[opt])
            all_conv[opt].append(conv[opt])

            if verbose:
                print(f'[Trial {i}] [{opt.upper()}] Converged: {bool(conv[opt])}, RMSE: {rmse[opt]:.8f}, steps: {k[opt]}')
        print()

    for opt in optimizers.keys():
        all_rmse_opt = torch.Tensor(all_rmse[opt])
        all_k_opt = torch.Tensor(all_k[opt])
        all_conv_opt = torch.Tensor(all_conv[opt])
        all_rmse_opt = all_rmse_opt[all_rmse_opt < torch.quantile(all_rmse_opt, 0.97)]
        mean_rmse = torch.nanmean(all_rmse_opt)
        mean_k = torch.nanmean(all_k_opt)
        mean_conv = torch.nanmean(all_conv_opt)

        if verbose:
            print(f'[{opt.upper()}]  Mean RMSE: {mean_rmse:.8f}, Mean steps: {mean_k:.2f}, Mean convergence: {mean_conv:.2f}')


if __name__ == '__main__':
    file = "../../nets/ieee118_186.mat"
    num_exp = 100
    verbose = True
    main(file, num_exp, verbose)
