import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from SE_np.net_preprocess.process_net_data import parse_ieee_mat, System
from SE_np.optimizers.NR_acpf_with_prior import NR_PF


def sample_data(sys, n=1000, mu=None, cov=None, **kwargs):
    if mu is not None and cov is not None:
        sampled_pq = np.zeros((n, sys.nb * 2))
        ang_mask = np.ones(sys.nb, dtype=bool)
        ang_mask[sys.slk_bus[0]] = False

        pq = np.where(sys.bus.bus_type.values == 1)[0]
        ang_idx_full = np.arange(0, sys.nb, dtype=int)
        volt_idx_full = sys.nb + np.arange(0, sys.nb, dtype=int)

        T_red_idx_full = ang_idx_full[ang_mask]
        V_red_idx_full = volt_idx_full[pq]

        red_full_idx = np.concatenate([T_red_idx_full, V_red_idx_full])
        sampled_pq[:, red_full_idx] = np.random.multivariate_normal(mean=mu, cov=cov, size=n)
        sampled_pl = sampled_pq[:, :sys.nb]
        sampled_ql = sampled_pq[:, sys.nb:]

        sampled_ql *= -1
        sampled_pl *= -1
        sampled_pg = np.zeros((n, sys.nb))
        sampled_qg = np.zeros((n, sys.nb))
        sampled_timeseries = None
    else:
        pl = pd.read_csv('data/time_series2/ieee118_186_Pl_timeseries.csv', index_col=0)
        pg = pd.read_csv('data/time_series2/ieee118_186_Pg_timeseries.csv', index_col=0)
        ql = pd.read_csv('data/time_series2/ieee118_186_Ql_timeseries.csv', index_col=0)
        qg = pd.read_csv('data/time_series2/ieee118_186_Qg_timeseries.csv', index_col=0)

        timeseries = pl.index.values
        sampled_timeseries = np.random.choice(timeseries, size=n)
        sampled_pl = pl.loc[sampled_timeseries].values
        sampled_pg = pg.loc[sampled_timeseries].values
        sampled_ql = ql.loc[sampled_timeseries].values
        sampled_qg = qg.loc[sampled_timeseries].values

    return sampled_pl, sampled_pg, sampled_ql, sampled_qg, sampled_timeseries


def EM(sys, n=1000, num_iters=100, tol=1e-5):
    mu_pq = np.load('../explore_prior/mu_pl.npy')
    cov_pq = np.load('../explore_prior/cov_pl.npy')
    Pl, Pg, Ql, Qg, sampled_timeseries = sample_data(sys, n, mu_pq, cov_pq)

    R = np.eye(sys.nb * 2) * 1e-5
    R_inv = np.linalg.inv(R)
    m = np.r_[np.zeros(sys.nb), np.ones(sys.nb)]
    x_init = np.c_[np.zeros(sys.nb), np.ones(sys.nb)]

    user = {'list': ['voltage'], 'stop': 1e-8, 'maxIter': 500}
    mu_T, mu_V = 0.45, 10
    Q = np.eye(sys.nb * 2)
    Q_inv = Q.copy()
    # Q_inv = -np.block([[mu_T * np.imag(sys.Ybus), np.zeros((sys.nb, sys.nb))],
    #                [np.zeros((sys.nb, sys.nb)), mu_V * np.imag(sys.Ybus)]])
    # Q = np.linalg.inv(Q_inv)
    reg_scale = 1
    for it in range(num_iters):
        Vcs = []
        for i in tqdm(range(n), desc=f'Power Flow Iteration {it}', colour='green', leave=False):
            curr_sys = sys = sys.copy()
            Vcs.append(NR_PF(curr_sys, np.c_[Pl[i], Ql[i]], np.c_[Pg[i], Qg[i]], x_init, user, m=m, Q=Q_inv, R=R_inv, reg_scale=reg_scale)['Vc'])

        Vcs = np.stack(Vcs)
        S = np.c_[np.angle(Vcs), np.abs(Vcs)]
        m_new = np.mean(S, axis=0)
        Q_new = np.cov(S, rowvar=False)
        L = np.linalg.cholesky(Q_new + 1e-10 * np.eye(len(Q)))
        L_inv = np.linalg.inv(L)
        # Q_new_inv = L_inv.T @ L_inv
        Q_new_inv = np.linalg.pinv(Q_new)
        plt.imshow(Q_new, cmap='gray')
        plt.show()
        plt.imshow(Q_new[:sys.nb, :sys.nb], cmap='gray')
        plt.show()
        plt.imshow(Q_new[sys.nb:, sys.nb:], cmap='gray')
        plt.show()
        delta_m = np.linalg.norm(m_new - m)
        delta_Q = np.linalg.norm(Q_new - Q)
        print(f'delta m = {delta_m}, delta_Q = {delta_Q}')
        if delta_m < tol and delta_Q < tol:
            break
        m, Q, Q_inv = m_new, Q_new, Q_new_inv
        # reg_scale = 1

    return m, Q

if __name__ == '__main__':
    file = '../../nets/ieee118_186.mat'

    data = parse_ieee_mat(file)
    system_data = data['data']['system']
    sys = System(system_data)

    m, Q = EM(sys)
    np.save('../explore_prior/mu_v.npy', m)
    np.save('../explore_prior/cov_v.npy', Q)
    print(m)
    print(Q)