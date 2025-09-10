import os

import numpy as np
import pandas as pd

from power_flow_ac.NR_pf import NR_PF


class DataGenerator:
    def __init__(self, Pg=None, Pl=None, Qg=None, Ql=None, data_dir='../data_parser/data/time_series/'):
        self.Pg = Pg if Pg is not None else pd.read_csv(
            f'{data_dir}/ieee118_186_Pg_timeseries.csv',
            index_col=0
        )
        self.Pl = Pl if Pl is not None else pd.read_csv(
            f'{data_dir}/ieee118_186_Pl_timeseries.csv',
            index_col=0
        )
        self.Qg = Qg if Qg is not None else pd.read_csv(
            f'{data_dir}/ieee118_186_Qg_timeseries.csv',
            index_col=0
        )
        self.Ql = Ql if Ql is not None else pd.read_csv(
            f'{data_dir}/ieee118_186_Ql_timeseries.csv',
            index_col=0
        )
        self.timeseries = self.Pl.index.values

        self.m = np.load(
            f'{data_dir}/mean.npy') if os.path.exists(
            f'{data_dir}/mean.npy') else None
        self.cov = np.load(
            f'{data_dir}/covariance.npy') if os.path.exists(
            f'{data_dir}/covariance.npy') else None

    def sample(self, sys, num_samples=1):
        sampled_timeseries = np.random.choice(self.timeseries, size=num_samples)
        Pl = self.Pl.loc[sampled_timeseries].values
        Pg = self.Pg.loc[sampled_timeseries].values
        Ql = self.Ql.loc[sampled_timeseries].values
        Qg = self.Qg.loc[sampled_timeseries].values

        x_init = sys.bus.loc[:, ['To', 'Vo']].values

        user = {'list': ['voltage'], 'stop': 1e-8, 'maxIter': 500}

        Vcs = []
        for i in range(num_samples):
            curr_sys = sys.copy()
            pf = NR_PF(curr_sys, np.c_[Pl[i], Ql[i]], np.c_[Pg[i], Qg[i]], x_init, user)
            Vcs.append(pf['Vc'])

        Vcs = np.stack(Vcs)
        if num_samples == 1:
            Vcs = Vcs.flatten()

        T, V = np.angle(Vcs), np.abs(Vcs)
        return T, V