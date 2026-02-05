import os

import torch
import pandas as pd
from tqdm import tqdm

from SE_torch.utils import init_start_point
from SE_torch.optimizers.NR_acpf import NR_PF
from SE_torch.PF_equations.PF_polar import H_AC as H_AC_polar
from SE_torch.PF_equations.PF_cartesian import H_AC as H_AC_cartesian


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
            v = torch.nonzero(mask_t, as_tuple=False).view(-1).to(torch.long)
        agg_meas_idx[k] = torch.arange(
            last_idx, last_idx + v.numel(), dtype=torch.long, device=device
        )
        last_idx += v.numel()

    return agg_meas_idx

def regenerate_PQ_numeric_types(
    sys,
    seed: int | None = None,
    load_sigma_frac: float = 0.12,
    pf_mean_by_type_id: dict | None = None,
    pf_sigma: float = 0.02,
    gen_noise_frac: float = 0.08,
    device: str | torch.device | None = None
):
    """
    Torch version. Returns (P_L, Q_L, Pg, Qg) as torch.float64 tensors on 'device'.
    Uses only pandas to read sys.bus; all computations in torch.
    """
    bus = sys.bus.copy()
    nb = sys.nb
    assert len(bus) == nb, "sys.nb and sys.bus length mismatch"

    Pbase_L = torch.as_tensor(bus['Pl'].fillna(0.0).values, device=device ,dtype=torch.get_default_dtype())
    if not torch.any(Pbase_L > 0):
        Pbase_L = torch.ones(nb, device=device)

    dP = torch.normal(mean=0.0, std=load_sigma_frac, size=(nb,), device=device, dtype=torch.get_default_dtype())
    P_L = torch.clamp(Pbase_L * (1.0 + dP), min=0.0)

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
    pf_nom[(pf_nom != pf_nom)] = 0.95
    min_pf = min(list(pf_mean_by_type_id.values()))
    max_pf = max(list(pf_mean_by_type_id.values()))
    pf_L = torch.normal(mean=pf_nom, std=pf_sigma)
    pf_L = torch.clamp(pf_L, min_pf, max_pf)

    Q_L = P_L * torch.tan(torch.arccos(pf_L))

    Qmin = torch.as_tensor(bus['Qmin'].to_numpy(float), device=device, dtype=torch.get_default_dtype())
    Qmax = torch.as_tensor(bus['Qmax'].to_numpy(float), device=device, dtype=torch.get_default_dtype())
    prev_Pg = torch.as_tensor(bus['Pg'].fillna(0.0).to_numpy(float), device=device, dtype=torch.get_default_dtype())

    finite_min = torch.isfinite(Qmin)
    finite_max = torch.isfinite(Qmax)
    gen_mask = (finite_min & finite_max) | (prev_Pg > 0)
    gen_idx = torch.nonzero(gen_mask, as_tuple=True)[0]

    Pg = torch.zeros(nb, device=device, dtype=torch.get_default_dtype())
    if gen_idx.numel() > 0:
        w = torch.clamp(prev_Pg[gen_idx], min=0.0)
        if w.sum() <= 1e-9:
            w = torch.ones_like(w)

        P_target = P_L.sum()
        Pg_raw = P_target * (w / w.sum())

        Pg_noise = torch.normal(mean=0.0, std=gen_noise_frac, size=Pg_raw.shape, device=device)
        Pg_gen = torch.clamp(Pg_raw + Pg_raw * Pg_noise, min=0.0)

        if Pg_gen.sum() > 1e-9:
            Pg_gen = Pg_gen * (P_target / Pg_gen.sum())

        Pg[gen_idx] = Pg_gen

    Qmin_eff = torch.where(gen_mask, torch.nan_to_num(Qmin, nan=0.0), torch.zeros_like(Qmin))
    Qmax_eff = torch.where(gen_mask, torch.nan_to_num(Qmax, nan=0.0), torch.zeros_like(Qmax))

    Qg = torch.zeros(nb, device=device)
    has_range = (Qmax_eff > Qmin_eff)
    mid = 0.5 * (Qmin_eff + Qmax_eff)
    Qg[has_range] = mid[has_range]

    Q_def = Q_L.sum() - Qg.sum()
    if torch.abs(Q_def) > 1e-9 and gen_idx.numel() > 0:
        room_up = torch.clamp(Qmax_eff - Qg, min=0.0)
        room_dn = torch.clamp(Qg - Qmin_eff, min=0.0)

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
        self.Pl = None
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

    def sample(self, sys, num_samples=1, random_flow=False, cart=False, seed=None, verbose=False, **kwargs):
        g = torch.Generator(device=self.device)
        if seed is not None:
            g.manual_seed(seed)
        pf_mean_by_type_id = kwargs.get('pf_mean_by_type_id', None)
        if random_flow:
            Pl, Ql, Pg, Qg = [], [], [], []
            for i in range(num_samples):
                Pli, Qli, Pgi, Qgi = regenerate_PQ_numeric_types(sys, device=self.device, seed=seed, pf_mean_by_type_id=pf_mean_by_type_id)
                Pl.append(Pli)
                Ql.append(Qli)
                Pg.append(Pgi)
                Qg.append(Qgi)

            Pl = torch.stack(Pl).to(dtype=torch.get_default_dtype())
            Ql = torch.stack(Ql).to(dtype=torch.get_default_dtype())
            Pg = torch.stack(Pg).to(dtype=torch.get_default_dtype())
            Qg = torch.stack(Qg).to(dtype=torch.get_default_dtype())
        else:
            if self.Pl is None:
                self.load_flow_from_dir()
            n_rows = self.Pl.shape[0]
            row_idx = torch.randint(low=0, high=n_rows, size=(num_samples,), generator=g, device=self.device)
            Pl = self.Pl.index_select(0, row_idx)
            Pg = self.Pg.index_select(0, row_idx)
            Ql = self.Ql.index_select(0, row_idx)
            Qg = self.Qg.index_select(0, row_idx)

        T0, V0 = init_start_point(sys)
        x_init = torch.stack([T0, V0], dim=1)

        user = {'list': ['voltage'], 'stop': 1e-8, 'maxIter': 500}

        Vcs = []
        for i in tqdm(range(Pl.shape[0]), desc="Generating data", colour='MAGENTA', disable=not verbose, leave=False):
            curr_sys = sys.copy()
            loads_i = torch.stack([Pl[i], Ql[i]], dim=1)
            gens_i  = torch.stack([Pg[i], Qg[i]], dim=1)
            pf = NR_PF(curr_sys, loads_i, gens_i, x_init, user, device=self.device)
            Vcs.append(pf['Vc'])

        Vcs = torch.stack(Vcs, dim=0)
        if num_samples == 1:
            Vcs = Vcs.view(-1)

        if cart:
            return Vcs.real, Vcs.imag

        T = torch.angle(Vcs).to(torch.get_default_dtype())
        V = torch.abs(Vcs).to(torch.get_default_dtype())

        return T, V

    def generate_measurements(self, sys, branch, T_true=None, V_true=None, device=None, **kwargs):
        if T_true is None or V_true is None:
            T_true, V_true = self.sample(sys, random_flow=True, **kwargs)
        T_true = T_true.to(device=device, dtype=torch.get_default_dtype())
        V_true = V_true.to(device=device, dtype=torch.get_default_dtype())
        v_real, v_imag = V_true * torch.cos(T_true), V_true * torch.sin(T_true)
        Vc_true = torch.cat([v_real, v_imag], dim=-1)

        keep_nans = bool(kwargs.get("keep_nans", False))
        sample_cfg = kwargs.get("sample", 1.0)

        gen = torch.Generator(device=device)
        gen.manual_seed(666)

        def _p_for_type(mtype: str) -> float:
            if isinstance(sample_cfg, dict):
                p = float(sample_cfg.get(mtype, 1.0))
            else:
                p = float(sample_cfg)
            return max(0.0, min(1.0, p))

        def _bernoulli_mask(n: int, p: float) -> torch.Tensor:
            if n <= 0:
                return torch.empty((0,), device=device, dtype=torch.bool)
            return torch.bernoulli(torch.full((n,), p, device=device), generator=gen).to(torch.bool)

        meas_idx = {}
        bus_mask_all = torch.ones(len(sys.bus), device=device, dtype=torch.bool)
        nbr = len(branch.i)
        half = nbr // 2
        meas_masks = {}
        if kwargs.get("flow"):
            PQf_mask = torch.cat(
                [torch.ones(half, device=device), torch.zeros(half, device=device)],
                dim=0
            ).to(torch.bool)
            meas_masks["Pf"] = PQf_mask
            meas_masks["Qf"] = PQf_mask
            if not keep_nans:
                p = _p_for_type("Pf")
                samp = _bernoulli_mask(int(PQf_mask.sum().item()), p)
                PQf_mask_s = PQf_mask.clone()
                PQf_mask_s[PQf_mask] = samp
                meas_idx["Pf_idx"] = PQf_mask_s
                meas_idx["Qf_idx"] = PQf_mask_s
            else:
                meas_idx["Pf_idx"] = PQf_mask
                meas_idx["Qf_idx"] = PQf_mask

        if kwargs.get("injection"):
            meas_masks["Pi"] = bus_mask_all
            meas_masks["Qi"] = bus_mask_all
            if not keep_nans:
                p = _p_for_type("Pi")
                samp = _bernoulli_mask(int(bus_mask_all.sum().item()), p)
                inj_mask = bus_mask_all.clone()
                inj_mask[bus_mask_all] = samp
                meas_idx["Pi_idx"] = inj_mask
                meas_idx["Qi_idx"] = inj_mask
            else:
                meas_idx["Pi_idx"] = bus_mask_all
                meas_idx["Qi_idx"] = bus_mask_all

        if kwargs.get("voltage"):
            meas_masks["Vm"] = bus_mask_all
            if not keep_nans:
                p = _p_for_type("Vm")
                samp = _bernoulli_mask(int(bus_mask_all.sum().item()), p)
                vm_mask = bus_mask_all.clone()
                vm_mask[bus_mask_all] = samp
                meas_idx["Vm_idx"] = vm_mask
            else:
                meas_idx["Vm_idx"] = bus_mask_all

        if kwargs.get("current"):
            cm_mask_all = torch.ones(len(branch.i), device=device, dtype=torch.bool)
            meas_masks["Cm"] = cm_mask_all
            if not keep_nans:
                p = _p_for_type("Cm")
                samp = _bernoulli_mask(int(cm_mask_all.sum().item()), p)
                cm_mask = cm_mask_all.clone()
                cm_mask[cm_mask_all] = samp
                meas_idx["Cm_idx"] = cm_mask
            else:
                meas_idx["Cm_idx"] = cm_mask_all

        meas_types = ["Pf", "Qf", "Cm", "Pi", "Qi", "Vm"]
        agg_meas_idx = aggregate_meas_idx(meas_idx, meas_types)

        h_ac_cart = H_AC_cartesian(sys, branch, meas_idx)
        h_ac_polar = H_AC_polar(sys, branch, meas_idx)

        z = torch.as_tensor(h_ac_polar.estimate(T_true, V_true), device=device)

        var = torch.ones(z.numel(), device=device)

        if kwargs.get("noise"):
            pieces = []
            for mtype in meas_types:
                count = len(agg_meas_idx[mtype])
                sigma2 = float(kwargs.get(f"{mtype}_noise", 1.0))
                pieces.append(torch.full((count,), sigma2, device=device))
            var = torch.cat(pieces, dim=0)

            if keep_nans:
                sample_pieces = []
                for mtype in meas_types:
                    if mtype == "Qf" or mtype == "Qi":
                        continue
                    mask = meas_masks.get(mtype)
                    if mask is None:
                        continue
                    mask_s = mask.clone()
                    count = int(mask.sum().item())
                    if count == 0:
                        continue
                    p = _p_for_type(mtype)
                    samp = _bernoulli_mask(count, p)
                    mask_s[mask] = samp
                    sample_pieces.append(samp)
                    if mtype == "Pf" or mtype == "Pi":
                        sample_pieces.append(samp)

                sample = torch.cat(sample_pieces, dim=0).to(torch.int) if len(sample_pieces) else torch.empty((0,), device=device)
                var = (var * sample) + (1 - sample) * 10
            noise = torch.normal(mean=torch.zeros_like(var), std=torch.sqrt(var))
            z = z + noise

        return z, var, meas_idx, agg_meas_idx, h_ac_cart, h_ac_polar, T_true, V_true, Vc_true