import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from init_net.process_net_data import parse_ieee_mat, System
from power_flow_ac.NR_pf_with_prior import NR_PF


def sample_data(n=1000):
    pl = pd.read_csv('../data_parser/data/time_series2/ieee118_186_Pl_timeseries.csv', index_col=0)
    pg = pd.read_csv('../data_parser/data/time_series2/ieee118_186_Pg_timeseries.csv', index_col=0)
    ql = pd.read_csv('../data_parser/data/time_series2/ieee118_186_Ql_timeseries.csv', index_col=0)
    qg = pd.read_csv('../data_parser/data/time_series2/ieee118_186_Qg_timeseries.csv', index_col=0)

    timeseries = pl.index.values
    sampled_timeseries = np.random.choice(timeseries, size=n)
    sampled_pl = pl.loc[sampled_timeseries].values
    sampled_pg = pg.loc[sampled_timeseries].values
    sampled_ql = ql.loc[sampled_timeseries].values
    sampled_qg = qg.loc[sampled_timeseries].values

    return sampled_pl, sampled_pg, sampled_ql, sampled_qg, sampled_timeseries


def EM(sys, n=100, num_iters=6, tol=1e-5):
    Pl, Pg, Ql, Qg, sampled_timeseries = sample_data(n)
    x_init = sys.bus.loc[:, ['To', 'Vo']].values


    user = {'list': ['voltage'], 'stop': 1e-8, 'maxIter': 500}
    R = np.eye(sys.nb * 2) * 1e-5
    R_inv = np.linalg.inv(R)
    m = np.r_[np.zeros(sys.nb), np.ones(sys.nb)]
    mu_T, mu_V = 0.45, 10
    Q_inv = -np.block([[mu_T * np.imag(sys.Ybus), np.zeros((sys.nb, sys.nb))],
                   [np.zeros((sys.nb, sys.nb)), mu_V * np.imag(sys.Ybus)]])
    Q = np.linalg.inv(Q_inv)
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
        Q_new_inv = L_inv.T @ L_inv
        plt.imshow(Q_new[:sys.nb, :sys.nb], cmap='gray')
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
    file = '../nets/ieee118_186.mat'

    data = parse_ieee_mat(file)
    system_data = data['data']['system']
    sys = System(system_data)

    m, Q = EM(sys)
    np.save('../data_parser/data/time_series2/mean.npy', m)
    np.save('../data_parser/data/time_series2/covariance.npy', Q)
    print(m)
    print(Q)