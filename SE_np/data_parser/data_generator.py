import os

import numpy as np
import pandas as pd

from SE_np.optimizers.NR_acpf import NR_PF
from SE_np.utils import init_start_point


def regenerate_PQ_numeric_types(
    sys,
    seed: int = None,
    load_sigma_frac: float = 0.12,      # ~12% std on P_load
    pf_mean_by_type_id: dict = None,    # {1:..., 2:..., 3:...} -> mean pf
    pf_sigma: float = 0.02,             # pf std
    gen_noise_frac: float = 0.08        # ~8% noise on P_gen
):
    rng = np.random.default_rng(seed)
    bus = sys.bus.copy()

    nb = sys.nb
    assert len(bus) == nb, "sys.nb and sys.bus length mismatch"

    Pbase_L = bus['Pl'].fillna(0.0).to_numpy(dtype=float)
    if not np.any(Pbase_L):
        Pbase_L = np.ones(nb) * 1.0  # small synthetic base

    dP = rng.normal(0.0, load_sigma_frac, size=nb)
    P_L = (Pbase_L * (1.0 + dP)).clip(min=0.0)

    if pf_mean_by_type_id is None:
        pf_mean_by_type_id = {
            1: 0.95,   # PQ (load)
            2: 0.98,   # PV
            3: 0.99    # Slack
        }

    types_id = bus['bus_type'].astype(int).to_numpy()
    pf_nom = np.array([pf_mean_by_type_id.get(t, 0.95) for t in types_id], dtype=float)
    pf_L = np.clip(rng.normal(pf_nom, pf_sigma, size=nb), 0.80, 0.999)

    Q_L = P_L * np.tan(np.arccos(pf_L))

    Qmin = bus['Qmin'].to_numpy(dtype=float)
    Qmax = bus['Qmax'].to_numpy(dtype=float)
    prev_Pg = bus['Pg'].fillna(0.0).to_numpy(dtype=float)

    gen_mask = (np.isfinite(Qmin) & np.isfinite(Qmax)) | (prev_Pg > 0)
    gen_idx = np.where(gen_mask)[0]
    Pg = np.zeros(nb, dtype=float)

    if gen_idx.size > 0:
        w = prev_Pg[gen_idx].clip(min=0.0)
        if w.sum() <= 1e-9:
            w = np.ones_like(w)

        P_target = P_L.sum()
        Pg_raw = P_target * (w / w.sum())

        Pg_noise = Pg_raw * rng.normal(0.0, gen_noise_frac, size=gen_idx.size)
        Pg_gen = np.maximum(0.0, Pg_raw + Pg_noise)

        if Pg_gen.sum() > 1e-9:
            Pg_gen *= (P_target / Pg_gen.sum())

        Pg[gen_idx] = Pg_gen

    Qmin_eff = np.where(gen_mask, np.nan_to_num(Qmin, nan=0.0), 0.0)
    Qmax_eff = np.where(gen_mask, np.nan_to_num(Qmax, nan=0.0), 0.0)

    Qg = np.zeros(nb, dtype=float)
    has_range = (Qmax_eff > Qmin_eff)
    Qg[has_range] = 0.5 * (Qmin_eff[has_range] + Qmax_eff[has_range])

    Q_def = Q_L.sum() - Qg.sum()  # >0 → need more lagging (+Q)
    if abs(Q_def) > 1e-9 and gen_idx.size > 0:
        room_up = (Qmax_eff - Qg).clip(min=0.0)  # can go more +Q
        room_dn = (Qg - Qmin_eff).clip(min=0.0)  # can go more -Q

        if Q_def > 0 and room_up.sum() > 1e-12:
            w = room_up / room_up.sum()
            Qg += Q_def * w
            Qg = np.minimum(Qg, Qmax_eff)
        elif Q_def < 0 and room_dn.sum() > 1e-12:
            w = room_dn / room_dn.sum()
            Qg += Q_def * w
            Qg = np.maximum(Qg, Qmin_eff)

    Qg[~gen_mask] = 0.0

    return P_L, Q_L, Pg, Qg


class DataGenerator:
    def __init__(self, Pg=None, Pl=None, Qg=None, Ql=None, data_dir='../data_parser/data/time_series2/'):
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

    def sample(self, sys, num_samples=1, random_flow=False):
        if random_flow:
            Pl, Ql, Pg, Qg = regenerate_PQ_numeric_types(sys)
            Pl = Pl.reshape(1, -1)
            Ql = Ql.reshape(1, -1)
            Pg = Pg.reshape(1, -1)
            Qg = Qg.reshape(1, -1)
        else:
            sampled_timeseries = np.random.choice(self.timeseries, size=num_samples)
            Pl = self.Pl.loc[sampled_timeseries].values
            Pg = self.Pg.loc[sampled_timeseries].values
            Ql = self.Ql.loc[sampled_timeseries].values
            Qg = self.Qg.loc[sampled_timeseries].values


        T0, V0 = init_start_point(sys, how='flat')
        x_init = np.c_[T0, V0]
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