# torch_data_generator.py
import os
import torch
import pandas as pd

from SE_torch.optimizers.NR_acpf import NR_PF
from SE_torch.PF_equations.PF_polar import H_AC as H_AC_polar
from SE_torch.PF_equations.PF_cartesian import H_AC as H_AC_cartesian

torch.set_default_dtype(torch.float64)


def aggregate_meas_idx(meas_idx: dict, meas_types: list[str], device=None):
    agg_meas_idx = {}
    last_idx = 0

    for k in meas_types:
        mask = meas_idx.get(f"{k}_idx", None)
        if mask is None:
            v = torch.tensor([], dtype=torch.long, device=device)
        else:
            mask_t = (
                mask.to(device=device)
                if isinstance(mask, torch.Tensor)
                else torch.as_tensor(mask, dtype=torch.bool, device=device)
            )
            # indices of True entries
            v = torch.nonzero(mask_t, as_tuple=False).view(-1).to(torch.long)
        agg_meas_idx[k] = torch.arange(
            last_idx, last_idx + v.numel(), dtype=torch.long, device=device
        )
        last_idx += v.numel()

    return agg_meas_idx

def regenerate_PQ_numeric_types(
    sys,
    seed: int | None = None,
    load_sigma_frac: float = 0.12,      # ~12% std on P_load
    pf_mean_by_type_id: dict | None = None,  # {1:..., 2:..., 3:...} -> mean pf
    pf_sigma: float = 0.02,             # pf std
    gen_noise_frac: float = 0.08,       # ~8% noise on P_gen
    device: str | torch.device | None = None
):
    """
    Torch version. Returns (P_L, Q_L, Pg, Qg) as torch.float64 tensors on 'device'.
    Uses only pandas to read sys.bus; all computations in torch.
    """
    g = torch.Generator(device=device)
    if seed is not None:
        g.manual_seed(seed)

    bus = sys.bus.copy()
    nb = sys.nb
    assert len(bus) == nb, "sys.nb and sys.bus length mismatch"

    # Base active load
    Pbase_L = torch.as_tensor(bus['Pl'].fillna(0.0).to_numpy(float), device=device)
    if not torch.any(Pbase_L > 0):
        Pbase_L = torch.ones(nb, device=device)

    # Random multiplicative noise on P
    dP = torch.normal(mean=0.0, std=load_sigma_frac, size=(nb,), generator=g, device=device)
    P_L = torch.clamp(Pbase_L * (1.0 + dP), min=0.0)

    # PF per numeric bus_type
    if pf_mean_by_type_id is None:
        pf_mean_by_type_id = {
            1: 0.95,  # PQ
            2: 0.98,  # PV
            3: 0.99   # Slack
        }
    types_id = torch.as_tensor(bus['bus_type'].to_numpy(int), device=device)
    pf_nom = torch.empty(nb, device=device)
    for tval, mean_pf in pf_mean_by_type_id.items():
        pf_nom[types_id == int(tval)] = float(mean_pf)
    # default for missing types
    pf_nom[(pf_nom != pf_nom)] = 0.95  # NaNs to 0.95 (unlikely)

    pf_L = torch.normal(mean=pf_nom, std=pf_sigma, generator=g)
    pf_L = torch.clamp(pf_L, 0.80, 0.999)

    # Reactive load from pf (lagging → +Q)
    # Q = P * tan(acos(pf))
    Q_L = P_L * torch.tan(torch.arccos(pf_L))

    # Generator masks & previous Pg
    Qmin = torch.as_tensor(bus['Qmin'].to_numpy(float), device=device)
    Qmax = torch.as_tensor(bus['Qmax'].to_numpy(float), device=device)
    prev_Pg = torch.as_tensor(bus['Pg'].fillna(0.0).to_numpy(float), device=device)

    finite_min = torch.isfinite(Qmin)
    finite_max = torch.isfinite(Qmax)
    gen_mask = (finite_min & finite_max) | (prev_Pg > 0)
    gen_idx = torch.nonzero(gen_mask, as_tuple=True)[0]

    Pg = torch.zeros(nb, device=device)
    if gen_idx.numel() > 0:
        w = torch.clamp(prev_Pg[gen_idx], min=0.0)
        if w.sum() <= 1e-9:
            w = torch.ones_like(w)

        P_target = P_L.sum()
        Pg_raw = P_target * (w / w.sum())

        Pg_noise = torch.normal(mean=0.0, std=gen_noise_frac, size=Pg_raw.shape, generator=g, device=device)
        Pg_gen = torch.clamp(Pg_raw + Pg_raw * Pg_noise, min=0.0)

        if Pg_gen.sum() > 1e-9:
            Pg_gen = Pg_gen * (P_target / Pg_gen.sum())

        Pg[gen_idx] = Pg_gen

    # Reactive gen limits and balancing
    Qmin_eff = torch.where(gen_mask, torch.nan_to_num(Qmin, nan=0.0), torch.zeros_like(Qmin))
    Qmax_eff = torch.where(gen_mask, torch.nan_to_num(Qmax, nan=0.0), torch.zeros_like(Qmax))

    Qg = torch.zeros(nb, device=device)
    has_range = (Qmax_eff > Qmin_eff)
    mid = 0.5 * (Qmin_eff + Qmax_eff)
    Qg[has_range] = mid[has_range]

    Q_def = Q_L.sum() - Qg.sum()  # >0 → need more lagging (+Q)
    if torch.abs(Q_def) > 1e-9 and gen_idx.numel() > 0:
        room_up = torch.clamp(Qmax_eff - Qg, min=0.0)  # can go more +Q
        room_dn = torch.clamp(Qg - Qmin_eff, min=0.0)  # can go more -Q

        if Q_def > 0 and room_up.sum() > 1e-12:
            w = room_up / room_up.sum()
            Qg = torch.minimum(Qg + Q_def * w, Qmax_eff)
        elif Q_def < 0 and room_dn.sum() > 1e-12:
            w = room_dn / room_dn.sum()
            Qg = torch.maximum(Qg + Q_def * w, Qmin_eff)

    Qg[~gen_mask] = 0.0

    return P_L, Q_L, Pg, Qg


class DataGenerator:
    def __init__(self, device: str | torch.device | None = None):
        self.device = device
        self.Pl = None  # torch.Tensor [T, nb]
        self.Ql = None
        self.Pg = None
        self.Qg = None
        self.timeseries = None
        self.m = None
        self.cov = None

    def load_flow_from_dir(self, data_dir='../data_parser/data/time_series2/'):
        # Load CSVs with pandas, convert to torch tensors
        Pg_df = pd.read_csv(f'{data_dir}/ieee118_186_Pg_timeseries.csv', index_col=0)
        Pl_df = pd.read_csv(f'{data_dir}/ieee118_186_Pl_timeseries.csv', index_col=0)
        Qg_df = pd.read_csv(f'{data_dir}/ieee118_186_Qg_timeseries.csv', index_col=0)
        Ql_df = pd.read_csv(f'{data_dir}/ieee118_186_Ql_timeseries.csv', index_col=0)

        # ensure consistent index order
        self.timeseries = Pg_df.index.to_numpy()

        self.Pg = torch.as_tensor(Pg_df.values, device=self.device, dtype=torch.get_default_dtype())
        self.Pl = torch.as_tensor(Pl_df.values, device=self.device, dtype=torch.get_default_dtype())
        self.Qg = torch.as_tensor(Qg_df.values, device=self.device, dtype=torch.get_default_dtype())
        self.Ql = torch.as_tensor(Ql_df.values, device=self.device, dtype=torch.get_default_dtype())

        mean_path = f'{data_dir}/mean.npy'
        cov_path  = f'{data_dir}/covariance.npy'
        if os.path.exists(mean_path):
            self.m = torch.from_numpy(__import__("numpy").load(mean_path)).to(self.device)
        if os.path.exists(cov_path):
            self.cov = torch.from_numpy(__import__("numpy").load(cov_path)).to(self.device)

    def sample(self, sys, num_samples: int = 1, random_flow: bool = False, seed: int | None = None):
        g = torch.Generator(device=self.device)
        if seed is not None:
            g.manual_seed(seed)

        if random_flow:
            Pl, Ql, Pg, Qg = regenerate_PQ_numeric_types(sys, device=self.device, seed=seed)
            Pl = Pl.view(1, -1)
            Ql = Ql.view(1, -1)
            Pg = Pg.view(1, -1)
            Qg = Qg.view(1, -1)
        else:
            if self.Pl is None:
                self.load_flow_from_dir()
            n_rows = self.Pl.shape[0]
            row_idx = torch.randint(low=0, high=n_rows, size=(num_samples,), generator=g, device=self.device)
            Pl = self.Pl.index_select(0, row_idx)
            Pg = self.Pg.index_select(0, row_idx)
            Ql = self.Ql.index_select(0, row_idx)
            Qg = self.Qg.index_select(0, row_idx)

        x_init = torch.stack([torch.zeros(sys.nb, device=self.device),
                              torch.ones(sys.nb, device=self.device)], dim=1)

        user = {'list': ['voltage'], 'stop': 1e-8, 'maxIter': 500}

        Vcs = []
        for i in range(Pl.shape[0]):
            curr_sys = sys.copy()
            loads_i = torch.stack([Pl[i], Ql[i]], dim=1)
            gens_i  = torch.stack([Pg[i], Qg[i]], dim=1)
            pf = NR_PF(curr_sys, loads_i, gens_i, x_init, user, device=self.device)
            Vcs.append(pf['Vc'])

        Vcs = torch.stack(Vcs, dim=0)
        if num_samples == 1:
            Vcs = Vcs.view(-1)

        T = torch.angle(Vcs).to(torch.get_default_dtype())
        V = torch.abs(Vcs).to(torch.get_default_dtype())
        return T, V

    def generate_measurements(self, sys, branch, device=None, **kwargs):

        f64 = torch.float64
        c128 = torch.complex128
        sys.slk_bus = [sys.slk_bus[0], 0, 1]
        T_true, V_true = self.sample(sys, random_flow=True)
        T_true = T_true.to(device=device, dtype=f64)
        V_true = V_true.to(device=device, dtype=f64)

        Vc_true = V_true.to(f64) * torch.exp(1j * T_true.to(f64))
        Vc_true = Vc_true.to(dtype=c128, device=device)

        meas_idx = {}
        if kwargs.get('flow'):
            nbr = len(branch.i)
            half = nbr // 2
            Pf_mask = torch.cat([
                torch.ones(half, dtype=torch.bool, device=device),
                torch.zeros(half, dtype=torch.bool, device=device)
            ], dim=0)
            Qf_mask = Pf_mask.clone()
            meas_idx['Pf_idx'] = Pf_mask
            meas_idx['Qf_idx'] = Qf_mask

        if kwargs.get('injection'):
            meas_idx['Pi_idx'] = torch.ones(len(sys.bus), dtype=torch.bool, device=device)
            meas_idx['Qi_idx'] = torch.ones(len(sys.bus), dtype=torch.bool, device=device)

        if kwargs.get('voltage'):
            meas_idx['Vm_idx'] = torch.ones(len(sys.bus), dtype=torch.bool, device=device)

        if kwargs.get('current'):
            meas_idx['Cm_idx'] = torch.ones(len(branch.i), dtype=torch.bool, device=device)

        meas_types = ['Pf', 'Qf', 'Cm', 'Pi', 'Qi', 'Vm']

        agg_meas_idx = aggregate_meas_idx(meas_idx, meas_types)

        h_ac_cart = H_AC_cartesian(sys, branch, meas_idx)
        h_ac_polar = H_AC_polar(sys, branch, meas_idx)

        z = h_ac_polar.estimate(V=V_true, T=T_true)
        z = torch.as_tensor(z, device=device, dtype=f64)

        var = torch.ones(z.numel(), device=device, dtype=f64)

        if kwargs.get('noise'):
            pieces = []
            for mtype in meas_types:
                count = len(agg_meas_idx[mtype])
                sigma2 = float(kwargs.get(f'{mtype}_noise', 1.0))
                pieces.append(torch.full((count,), sigma2, device=device, dtype=f64))
            var = torch.cat(pieces, dim=0)

            noise = torch.normal(
                mean=torch.zeros_like(var),
                std=torch.sqrt(var)
            )
            z = z + noise

        return z, var, meas_idx, agg_meas_idx, h_ac_cart, h_ac_polar, T_true, V_true, Vc_true