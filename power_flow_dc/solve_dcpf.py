import numpy as np
from scipy.sparse import coo_array


def generate_Ybus_dc(bus_data, branch_data):
    inv_tij_xij = 1 / (branch_data['tij'].values * branch_data['xij'].values)

    nb = len(bus_data)
    nbr = len(branch_data)

    idx_branch = np.array(branch_data.index)
    idx_from, idx_to = branch_data[['idx_from', 'idx_to']].values.T

    Ai = coo_array((np.ones_like(idx_branch), [idx_branch, idx_from]), (nbr, nb)).toarray()
    Ai -= coo_array((np.ones_like(idx_branch), [idx_branch, idx_to]), (nbr, nb)).toarray()

    Yi = coo_array((inv_tij_xij, [idx_branch, idx_from]), (nbr, nb)).toarray()
    Yi -= coo_array((inv_tij_xij, [idx_branch, idx_to]), (nbr, nb)).toarray()

    Ybus = Ai.T @ Yi

    bus_data['Psh'] = -Ai.T @ (inv_tij_xij * branch_data['fij'].values)
    branch_data['inv_tij_xij'] = inv_tij_xij
    return Ybus, bus_data, branch_data


class Ai:
    def __init__(self, ai_idx, z, v, bus_data, Ybus):
        self.on = ai_idx
        self.N = np.count_nonzero(self.on)
        self.z = z
        self.v = v

        self.bus = bus_data.loc[self.on, 'idx_bus'].values.astype(int)

        self.H =  Ybus[self.bus]
        self.b = self.z - bus_data.loc[self.on, 'Psh'].values
        self.W = np.diag(1 / self.v)

class Va:
    def __init__(self, va_idx, z, v, bus_data, ts):
        self.on = va_idx
        self.N = np.count_nonzero(self.on)
        self.z = z
        self.v = v

        self.bus = bus_data.loc[self.on, 'idx_bus'].values.astype(int)

        nb = len(bus_data)
        n = np.arange(self.N)

        self.H =  coo_array((np.ones(self.N), [n, self.bus]), (self.N, nb)).toarray()
        self.b = self.z - ts[self.on]
        self.W = np.diag(1 / self.v)


class Af:
    def __init__(self, af_idx, z, v, bra, nb):
        self.on = af_idx
        self.N = np.count_nonzero(self.on)
        self.z = z
        self.v = v
        self.fro = bra.i[self.on].astype(int)
        self.to = bra.j[self.on].astype(int)

        n = np.arange(self.N)
        self.H = coo_array((bra.bij[self.on], [n, self.fro]), (self.N, nb)).toarray()
        self.b = self.z - bra.fij[self.on]
        self.W = np.diag(1 / self.v)


class H_DC:
    def __init__(self, sys, branch, Ybus, meas_idx, agg_meas_idx, z, v):
        ai_idx = np.zeros(sys.nb).astype(bool)
        va_idx = np.zeros(sys.nb).astype(bool)
        af_idx = np.zeros_like(branch.i).astype(bool)

        z_ai, v_ai = np.array([]), np.array([])
        z_af, v_af = np.array([]), np.array([])
        z_va, v_va = np.array([]), np.array([])

        if meas_idx.get('Pi_idx') is not None:
            ai_idx = meas_idx['Pi_idx']
            z_ai = z[agg_meas_idx['Pi']]
            v_ai = v[agg_meas_idx['Pi']]
        if meas_idx.get('Pf_idx') is not None:
            af_idx = meas_idx['Pf_idx']
            z_af = z[agg_meas_idx['Pf']]
            v_af = v[agg_meas_idx['Pf']]
        if meas_idx.get('Va_idx') is not None:
            va_idx = meas_idx['Va_idx']
            z_va = z[agg_meas_idx['Va']]
            v_va = v[agg_meas_idx['Va']]

        self.nb = sys.nb
        self.ai = Ai(ai_idx, z_ai, v_ai, sys.bus, Ybus)
        self.af = Af(af_idx, z_af, v_af, branch, self.nb)

        ts = sys.slk_bus[1] * np.ones(sys.nb)
        self.va = Va(va_idx, z_va, v_va, sys.bus, ts)


        self.H = np.r_[self.af.H, self.ai.H,self.va.H]
        self.b = np.r_[self.af.b, self.ai.b, self.va.b]
        self.W = np.diag(1. / np.r_[self.af.v, self.ai.v, self.va.v])



def DC_PF(slk_bus, h_dc):
    H = h_dc.H
    b = h_dc.b
    W = h_dc.W
    H = np.delete(H, slk_bus[0], axis=1)
    HT_W_H = H.T @ W @ H
    HT_W_b = H.T @ W @ b
    Va = np.linalg.lstsq(HT_W_H, HT_W_b, rcond=None)[0]
    Va = np.insert(Va, slk_bus[0], 0)
    Va += slk_bus[1]

    return Va

