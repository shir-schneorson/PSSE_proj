import numpy as np


class Pi:
    def __init__(self, Pi_idx, bus_data, Ybu, Yij, Yii):
        self.idx = Pi_idx
        self.bus = bus_data.loc[self.idx, 'idx_bus'].values.astype(int)
        i, j = np.nonzero(Ybu)
        idx1 = np.isin(i, self.bus)
        self.i = i[idx1]
        self.j = j[idx1]
        self.N = len(self.bus)

        Ybus = Ybu[self.bus]
        idx2 = np.nonzero(Ybus)
        self.ii, _ = idx2
        self.Gij = np.real(Ybus[idx2])
        self.Bij = np.imag(Ybus[idx2])

        self.Gii = np.real(Ybu[self.bus, self.bus])
        self.Bii = np.imag(Ybu[self.bus, self.bus])

        self.ij = self.i != self.j

        r, c = np.nonzero(Yij[self.bus])
        rr, cc = np.nonzero(Yii[self.bus])

        self.jci = np.r_[rr, r]
        self.jcj = np.r_[cc, c]


class Qi:
    def __init__(self, Qi_idx, bus_data, Ybu, Yij, Yii):
        self.idx = Qi_idx
        self.bus = bus_data.loc[self.idx, 'idx_bus'].values.astype(int)
        i, j = np.nonzero(Ybu)
        idx1 = np.isin(i, self.bus)
        self.i = i[idx1]
        self.j = j[idx1]
        self.N = len(self.bus)

        Ybus = Ybu[self.bus]
        idx2 = np.nonzero(Ybus)
        self.ii, _ = idx2
        self.Gij = np.real(Ybus[idx2])
        self.Bij = np.imag(Ybus[idx2])

        self.Gii = np.real(Ybu[self.bus, self.bus])
        self.Bii = np.imag(Ybu[self.bus, self.bus])

        self.ij = self.i != self.j

        r, c = np.nonzero(Yij[self.bus])
        rr, cc = np.nonzero(Yii[self.bus])

        self.jci = np.r_[rr, r]
        self.jcj = np.r_[cc, c]


class Cm:
    def __init__(self, Cm_idx, bus_data, bra):
        self.idx = Cm_idx
        self.i = bra.i[self.idx].astype(int)
        self.j = bra.j[self.idx].astype(int)
        self.N = len(self.i)

        self.A = (bra.tij[self.idx] ** 4 *
                  (bra.gij[self.idx] ** 2 +
                   (bra.bij[self.idx] + bra.bsi[self.idx]) ** 2))
        self.B = (bra.pij[self.idx] ** 2 *
                  (bra.gij[self.idx] ** 2 + bra.bij[self.idx] ** 2))
        self.C = (bra.tij[self.idx] ** 2 * bra.pij[self.idx] *
                  (bra.gij[self.idx] ** 2 + bra.bij[self.idx] *
                   (bra.bij[self.idx] + bra.bsi[self.idx])))
        self.D = bra.tij[self.idx] ** 2 * bra.pij[self.idx] * bra.gij[self.idx] * bra.bsi[self.idx]
        self.fij = bra.fij[self.idx]
        self.bus = bus_data['idx_bus'].values

        num = np.arange(self.N)
        self.jci = np.r_[num, num]
        self.jcj = np.r_[self.i, self.j]


class Pf:
    def __init__(self, Pf_idx, bra):
        self.idx = Pf_idx
        self.i = bra.i[self.idx].astype(int)
        self.j = bra.j[self.idx].astype(int)
        self.N = len(self.i)

        self.gij = bra.gij[self.idx]
        self.bij = bra.bij[self.idx]
        self.tgij = bra.tij[self.idx] ** 2 * self.gij
        self.pij = bra.pij[self.idx]
        self.fij = bra.fij[self.idx]

        num = np.arange(self.N)
        self.jci = np.r_[num, num]
        self.jcj = np.r_[self.i, self.j]


class Qf:
    def __init__(self, Qf_idx, bra):
        self.idx = Qf_idx
        self.i = bra.i[self.idx].astype(int)
        self.j = bra.j[self.idx].astype(int)
        self.N = len(self.i)

        self.gij = bra.gij[self.idx]
        self.bij = bra.bij[self.idx]
        self.bsi = bra.bsi[self.idx]
        self.tbij = bra.tij[self.idx] ** 2 * (self.bij + self.bsi)
        self.pij = bra.pij[self.idx]
        self.fij = bra.fij[self.idx]

        num = np.arange(self.N)
        self.jci = np.r_[num, num]
        self.jcj = np.r_[self.i, self.j]


class Vm:
    def __init__(self, Vm_idx, bus_data):
        self.idx = Vm_idx
        self.i = bus_data.loc[self.idx, 'idx_bus'].values.astype(int)
        self.N = len(self.i)