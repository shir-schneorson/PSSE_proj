import torch

from SE_torch.compose_measurment_torch import Pi, Qi, Pf, Qf, Cm, Vm


def injection_acse(V, T, p, q, nb):
    U = T[p.i] - T[p.j]
    Vi = V[p.i]
    Vj = V[p.j]
    Vb = V[p.bus]

    Tp1 = p.Gij * torch.cos(U) + p.Bij * torch.sin(U)
    Pi = Vb * (torch.sparse_coo_tensor(torch.vstack([p.ii, p.j]), Tp1, (p.N, nb)) @ V)

    Tp2 = -p.Gij * torch.sin(U) + p.Bij * torch.cos(U)
    Pi_Ti = Vb * (torch.sparse_coo_tensor(torch.vstack([p.ii, p.j]), Tp2, (p.N, nb)) @ V) - (Vb ** 2 * p.Bii)
    Pi_Tj = -Vi[p.ij] * Vj[p.ij] * Tp2[p.ij]

    Pi_Vi = (torch.sparse_coo_tensor(torch.vstack([p.ii, p.j]), Tp1, (p.N, nb)) @ V) + (Vb * p.Gii)
    Pi_Vj = Vi[p.ij] * Tp1[p.ij]

    J41 = torch.sparse_coo_tensor(torch.vstack([p.jci, p.jcj]), torch.cat([Pi_Ti, Pi_Tj]), (p.N, nb))
    J42 = torch.sparse_coo_tensor(torch.vstack([p.jci, p.jcj]), torch.cat([Pi_Vi, Pi_Vj]), (p.N, nb))

    U = T[q.i] - T[q.j]
    Vi = V[q.i]
    Vj = V[q.j]
    Vb = V[q.bus]

    Tq1 = q.Gij * torch.sin(U) - q.Bij * torch.cos(U)
    Qi = Vb * (torch.sparse_coo_tensor(torch.vstack([q.ii, q.j]), Tq1, (q.N, nb)) @ V)

    Tq2 = q.Gij * torch.cos(U) + q.Bij * torch.sin(U)
    Qi_Ti = Vb * (torch.sparse_coo_tensor(torch.vstack([q.ii, q.j]), Tq2, (q.N, nb)) @ V) - (Vb ** 2 * q.Gii)
    Qi_Tj = -Vi[q.ij] * Vj[q.ij] * Tq2[q.ij]

    Qi_Vi = (torch.sparse_coo_tensor(torch.vstack([q.ii, q.j]), Tq1, (q.N, nb)) @ V) - (Vb * q.Bii)
    Qi_Vj = Vi[q.ij] * Tq1[q.ij]

    J51 = torch.sparse_coo_tensor(torch.vstack([q.jci, q.jcj]), torch.cat([Qi_Ti, Qi_Tj]), (q.N, nb))
    J52 = torch.sparse_coo_tensor(torch.vstack([q.jci, q.jcj]), torch.cat([Qi_Vi, Qi_Vj]), (q.N, nb))

    Fi = torch.cat([Pi, Qi])
    Ji = torch.cat([torch.cat([J41, J42],dim=1), torch.cat([J51, J52], dim=1)], dim=0).to_dense()

    return Fi, Ji


def flow_acse(V, T, p, q, nb):
    U = T[p.i] - T[p.j] - p.fij
    Vi = V[p.i]
    Vj = V[p.j]

    Tp1 = p.gij * torch.cos(U) + p.bij * torch.sin(U)
    Pij = Vi ** 2 * p.tgij - p.pij * Vi * Vj * Tp1

    Tp2 = p.gij * torch.sin(U) - p.bij * torch.cos(U)
    Pij_Ti = p.pij * Vi * Vj * Tp2
    Pij_Tj = -Pij_Ti

    Pij_Vi = 2 * p.tgij * Vi - p.pij * Vj * Tp1
    Pij_Vj = -p.pij * Vi * Tp1

    J11 = torch.sparse_coo_tensor(torch.vstack([p.jci, p.jcj]), torch.cat([Pij_Ti, Pij_Tj]), (p.N, nb))
    J12 = torch.sparse_coo_tensor(torch.vstack([p.jci, p.jcj]), torch.cat([Pij_Vi, Pij_Vj]), (p.N, nb))

    U = T[q.i] - T[q.j] - q.fij
    Vi = V[q.i]
    Vj = V[q.j]

    Tq1 = q.gij * torch.sin(U) - q.bij * torch.cos(U)
    Qij = -q.tbij * Vi ** 2 - q.pij * Vi * Vj * Tq1

    Tq2 = q.gij * torch.cos(U) + q.bij * torch.sin(U)
    Qij_Ti = -q.pij * Vi * Vj * Tq2
    Qij_Tj = -Qij_Ti

    Qij_Vi = -2 * q.tbij * Vi - q.pij * Vj * Tq1
    Qij_Vj = -q.pij * Vi * Tq1

    J21 = torch.sparse_coo_tensor(torch.vstack([q.jci, q.jcj]), torch.cat([Qij_Ti, Qij_Tj]), (q.N, nb))
    J22 = torch.sparse_coo_tensor(torch.vstack([q.jci, q.jcj]), torch.cat([Qij_Vi, Qij_Vj]), (q.N, nb))

    Ff = torch.cat([Pij, Qij])
    Jf = torch.cat([torch.cat([J11, J12], dim=1), torch.cat([J21, J22], dim=1)], dim=0).to_dense()

    return Ff, Jf


def current_acse(V, T, c, nb):
    U = T[c.i] - T[c.j] - c.fij
    Vi = V[c.i]
    Vj = V[c.j]

    Tc1 = c.C * torch.cos(U) - c.D * torch.sin(U)
    Fc = torch.sqrt(c.A * Vi ** 2 + c.B * Vj ** 2 - 2 * Vi * Vj * Tc1)
    mask = Fc != 0

    Tc2 = c.C * torch.sin(U) + c.D * torch.cos(U)
    Iij_Ti = torch.zeros_like(Fc)
    Iij_Ti[mask] = (Vi[mask] * Vj[mask] * Tc2[mask]) / Fc[mask]
    Iij_Tj = -Iij_Ti

    Iij_Vi = torch.zeros_like(Fc)
    Iij_Vj = torch.zeros_like(Fc)
    Iij_Vi[mask] = (-Vj[mask] * Tc1[mask] + c.A[mask] * Vi[mask]) / Fc[mask]
    Iij_Vj[mask] = (-Vi[mask] * Tc1[mask] + c.B[mask] * Vj[mask]) / Fc[mask]

    J31 = torch.sparse_coo_tensor(torch.vstack([c.jci, c.jcj]), torch.cat([Iij_Ti, Iij_Tj]), (c.N, nb))
    J32 = torch.sparse_coo_tensor(torch.vstack([c.jci, c.jcj]), torch.cat([Iij_Vi, Iij_Vj]), (c.N, nb))

    Jc = torch.cat([J31, J32], dim=1).to_dense()

    return Fc, Jc


def voltage_acse(V, vm, nb):
    Fv = V[vm.i]

    V_V = torch.sparse_coo_tensor(torch.vstack([torch.arange(vm.N), vm.i]), torch.ones(vm.N), (vm.N, nb))
    V_T = torch.zeros((vm.N, nb))

    Jv = torch.cat([V_T, V_V.to_dense()], dim=1)

    return Fv, Jv


class H_AC:
    def __init__(self, sys, branch, meas_idx):
        Pi_idx = Qi_idx = Vm_idx = torch.zeros(sys.nb, dtype=torch.bool)
        Pf_idx = Qf_idx = Cm_idx = torch.zeros_like(branch.i, dtype=torch.bool)
        if meas_idx.get('Pi_idx') is not None:
            Pi_idx = meas_idx['Pi_idx']
        if meas_idx.get('Qi_idx') is not None:
            Qi_idx = meas_idx['Qi_idx']
        if meas_idx.get('Pf_idx') is not None:
            Pf_idx = meas_idx['Pf_idx']
        if meas_idx.get('Qf_idx') is not None:
            Qf_idx = meas_idx['Qf_idx']
        if meas_idx.get('Cm_idx') is not None:
            Cm_idx = meas_idx['Cm_idx']
        if meas_idx.get('Vm_idx') is not None:
            Vm_idx = meas_idx['Vm_idx']
        self.pi = Pi(Pi_idx, sys.bus, sys.Ybus, sys.Yij, sys.Yii)
        self.qi = Qi(Qi_idx, sys.bus, sys.Ybus, sys.Yij, sys.Yii)
        self.pf = Pf(Pf_idx, branch)
        self.qf = Qf(Qf_idx, branch)
        self.cm = Cm(Cm_idx, sys.bus, branch)
        self.vm = Vm(Vm_idx, sys.bus)
        self.nb = sys.nb

    def estimate(self, V, T):
        Ff, Jf = flow_acse(V, T, self.pf, self.qf, self.nb)
        Fi, Ji = injection_acse(V, T, self.pi, self.qi, self.nb)
        Fc, Jc = current_acse(V, T, self.cm, self.nb)
        Fv, Jv = voltage_acse(V, self.vm, self.nb)

        f = torch.cat([Ff, Fc, Fi, Fv])
        J = torch.cat([Jf, Jc, Ji, Jv])

        return f.float(), J.float()
