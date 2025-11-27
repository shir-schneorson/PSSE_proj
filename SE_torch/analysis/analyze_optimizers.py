import json

import torch

from SE_torch.PF_equations.PF_cartesian import H_AC as H_AC_cartesian
from SE_torch.data_generator import DataGenerator
from SE_torch.net_preprocess.process_net_data import parse_ieee_mat, System, Branch
from SE_torch.utils import square_mag, normalize_measurements, RMSE, init_start_point
from SE_torch.optimizers.GN_se import GN_se
from SE_torch.optimizers.LM_se import LM_se
from SE_torch.optimizers.FO_se import SGD_se, AdamW_se, Muon_se, LBFGS_se
from SE_torch.learn_prior.GNU_GNN.models import GNU_Model


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
    m, cov = torch.load('../learn_prior/datasets/mean.pt'), torch.load(
        '../learn_prior/datasets/cov.pt')

    GNU_GNN_config = json.load(open("../learn_prior/configs/GNU_GNN_config.json"))
    ckpt_path = f"../learn_prior/GNU_GNN/models/{GNU_GNN_config.get('ckpt_name')}"
    GNU_GNN_model = GNU_Model(**GNU_GNN_config)
    edge_index = torch.stack([branch.i, branch.j]).to(torch.long)
    GNU_GNN_model.load_state_dict(torch.load(ckpt_path))
    GNU_GNN_model.edge_index = edge_index
    GNU_GNN_model.slk_bus = sys.slk_bus

    optimizers = {'lbfgs_nf_8_0_128': LBFGS_se(verbose=False, use_prior=True,
                                               prior_config_path="../learn_prior/configs/NF_8_0_128.json"),
                  'lm_gs': LM_se(verbose=False, use_prior=True, m=m, Q=cov),
                  'gnu_gnn': GNU_GNN_model.optimize,
                  'gn': GN_se(verbose=False, max_iter=100)}

    all_rmse = {opt: [] for opt in optimizers.keys()}
    all_k = {opt: [] for opt in optimizers.keys()}
    all_conv = {opt: [] for opt in optimizers.keys()}

    for i in range(num_experiments):
        rmse, k, conv = run_experiment(data, sys, branch, data_generator, optimizers, **kwargs)
        for opt in optimizers.keys():
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
        # all_rmse_opt = all_rmse_opt[all_rmse_opt < torch.quantile(all_rmse_opt, 0.97)]
        mean_rmse = torch.nanmean(all_rmse_opt)
        mean_k = torch.nanmean(all_k_opt)
        mean_conv = torch.nanmean(all_conv_opt)

        if verbose:
            print(f'[{opt.upper()}]  Mean RMSE: {mean_rmse:.8f}, Mean steps: {mean_k:.2f}, Mean convergence: {mean_conv:.2f}')

def main():
    config = json.load(open("./configs/analyze_optimizer_config.json"))
    run_all_experiments(config)

if __name__ == '__main__':
    main()
