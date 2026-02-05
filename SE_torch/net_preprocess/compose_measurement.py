import numpy as np
import torch


class Pi:
    def __init__(self, Pi_idx, bus_data, Ybu, Yij, Yii):
        self.idx = Pi_idx
        self.bus = torch.tensor(bus_data.loc[self.idx.numpy(), 'idx_bus'].values).int()
        i, j = torch.nonzero(Ybu).T
        idx1 = torch.isin(i, self.bus)
        self.i = i[idx1]
        self.j = j[idx1]
        self.N = len(self.bus)

        Ybus = Ybu[self.bus]
        idx2 = torch.nonzero(Ybus).T
        self.ii, _ = idx2
        self.Gij = torch.real(Ybus[idx2[0], idx2[1]]).to(dtype=torch.get_default_dtype())
        self.Bij = torch.imag(Ybus[idx2[0], idx2[1]]).to(dtype=torch.get_default_dtype())

        self.Yii = Yii[self.bus, self.bus]
        self.Ybu = Ybu
        self.Gii = torch.real(Ybu[self.bus, self.bus]).to(dtype=torch.get_default_dtype())
        self.Bii = torch.imag(Ybu[self.bus, self.bus]).to(dtype=torch.get_default_dtype())

        self.ij = self.i != self.j

        r, c = torch.nonzero(Yij[self.bus]).T
        rr, cc = torch.nonzero(Yii[self.bus]).T

        self.jci = torch.tensor(np.r_[rr, r])
        self.jcj = torch.tensor(np.r_[cc, c])


class Qi:
    def __init__(self, Qi_idx, bus_data, Ybu, Yij, Yii):
        self.idx = Qi_idx
        self.bus = torch.tensor(bus_data.loc[self.idx.numpy(), 'idx_bus'].values).int()
        i, j = torch.nonzero(Ybu).T
        idx1 = np.isin(i, self.bus)
        self.i = i[idx1]
        self.j = j[idx1]
        self.N = len(self.bus)

        Ybus = Ybu[self.bus]
        idx2 = torch.nonzero(Ybus).T
        self.ii, _ = idx2
        self.Gij = torch.real(Ybus[idx2[0], idx2[1]]).to(dtype=torch.get_default_dtype())
        self.Bij = torch.imag(Ybus[idx2[0], idx2[1]]).to(dtype=torch.get_default_dtype())

        self.Yii = Yii[self.bus, self.bus]
        self.Ybu = Ybu
        self.Gii = torch.real(Ybu[self.bus, self.bus]).to(dtype=torch.get_default_dtype())
        self.Bii = torch.imag(Ybu[self.bus, self.bus]).to(dtype=torch.get_default_dtype())

        self.ij = self.i != self.j

        r, c = torch.nonzero(Yij[self.bus]).T
        rr, cc = torch.nonzero(Yii[self.bus]).T

        self.jci = torch.tensor(np.r_[rr, r])
        self.jcj = torch.tensor(np.r_[cc, c])


class Cm:
    def __init__(self, Cm_idx, bus_data, bra):
        self.idx = Cm_idx
        self.i = bra.i[self.idx].int()
        self.j = bra.j[self.idx].int()
        self.N = len(self.i)

        self.A = (bra.tij[self.idx] ** 4 *
                  (bra.gij[self.idx] ** 2 +
                   (bra.bij[self.idx] + bra.bsi[self.idx]) ** 2)).to(dtype=torch.get_default_dtype())
        self.B = (bra.pij[self.idx] ** 2 *
                  (bra.gij[self.idx] ** 2 + bra.bij[self.idx] ** 2)).to(dtype=torch.get_default_dtype())
        self.C = (bra.tij[self.idx] ** 2 * bra.pij[self.idx] *
                  (bra.gij[self.idx] ** 2 + bra.bij[self.idx] *
                   (bra.bij[self.idx] + bra.bsi[self.idx]))).to(dtype=torch.get_default_dtype())
        self.D = (bra.tij[self.idx] ** 2 * bra.pij[self.idx] * bra.gij[self.idx] * bra.bsi[self.idx]).to(dtype=torch.get_default_dtype())
        # self.A = torch.tensor(self.A)
        # self.B = torch.tensor(self.B)
        # self.C = torch.tensor(self.C)
        # self.D = torch.tensor(self.D)
        self.yij = bra.yij[self.idx]
        self.ysi = bra.ysi[self.idx]
        self.fij = bra.fij[self.idx]
        self.bus = torch.tensor(bus_data['idx_bus'].values).int()

        num = torch.arange(self.N)
        self.jci = torch.cat([num, num])
        self.jcj = torch.cat([self.i, self.j])


class Pf:
    def __init__(self, Pf_idx, bra):
        self.idx = Pf_idx
        self.i = bra.i[self.idx].int()
        self.j = bra.j[self.idx].int()
        self.N = len(self.i)

        self.yij = bra.yij[self.idx]
        self.ysi = bra.ysi[self.idx]
        self.gij = bra.gij[self.idx].to(dtype=torch.get_default_dtype())
        self.bij = bra.bij[self.idx].to(dtype=torch.get_default_dtype())
        self.tgij = (bra.tij[self.idx] ** 2 * self.gij).to(dtype=torch.get_default_dtype())
        self.pij = bra.pij[self.idx]
        self.fij = bra.fij[self.idx]

        num = torch.arange(self.N)
        self.jci = torch.cat([num, num])
        self.jcj = torch.cat([self.i, self.j])


class Qf:
    def __init__(self, Qf_idx, bra):
        self.idx = Qf_idx
        self.i = bra.i[self.idx].int()
        self.j = bra.j[self.idx].int()
        self.N = len(self.i)

        self.yij = bra.yij[self.idx]
        self.ysi = bra.ysi[self.idx]
        self.gij = bra.gij[self.idx].to(dtype=torch.get_default_dtype())
        self.bij = bra.bij[self.idx].to(dtype=torch.get_default_dtype())
        self.bsi = bra.bsi[self.idx].to(dtype=torch.get_default_dtype())
        self.tbij = (bra.tij[self.idx] ** 2 * (self.bij + self.bsi)).to(dtype=torch.get_default_dtype())
        self.pij = bra.pij[self.idx]
        self.fij = bra.fij[self.idx]

        num = torch.arange(self.N)
        self.jci = torch.cat([num, num])
        self.jcj = torch.cat([self.i, self.j])


class Vm:
    def __init__(self, Vm_idx, bus_data):
        self.idx = Vm_idx
        self.i = torch.tensor(bus_data.loc[self.idx.numpy(), 'idx_bus'].values).int()
        self.N = len(self.i)
