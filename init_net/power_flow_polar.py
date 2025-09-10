import numpy as np
from scipy.sparse import coo_array

from init_net.compose_meausrement import Pi, Qi, Pf, Qf, Cm, Vm


def injection_acse(V, T, p, q, nb):
    U = T[p.i] - T[p.j]
    Vi = V[p.i]
    Vj = V[p.j]
    Vb = V[p.bus]

    Tp1 = p.Gij * np.cos(U) + p.Bij * np.sin(U)
    Pi = Vb * (coo_array((Tp1, [p.ii, p.j]), (p.N, nb)).toarray() @ V)

    Tp2 = -p.Gij * np.sin(U) + p.Bij * np.cos(U)
    Pi_Ti = Vb * (coo_array((Tp2, [p.ii, p.j]), (p.N, nb)).toarray() @ V) - (Vb ** 2 * p.Bii)
    Pi_Tj = -Vi[p.ij] * Vj[p.ij] * Tp2[p.ij]

    Pi_Vi = (coo_array((Tp1, [p.ii, p.j]), (p.N, nb)).toarray() @ V) + (Vb * p.Gii)
    Pi_Vj = Vi[p.ij] * Tp1[p.ij]

    J41 = coo_array((np.r_[Pi_Ti, Pi_Tj], [p.jci, p.jcj]), (p.N, nb)).toarray()
    J42 = coo_array((np.r_[Pi_Vi, Pi_Vj], [p.jci, p.jcj]), (p.N, nb)).toarray()

    U = T[q.i] - T[q.j]
    Vi = V[q.i]
    Vj = V[q.j]
    Vb = V[q.bus]

    Tq1 = q.Gij * np.sin(U) - q.Bij * np.cos(U)
    Qi = Vb * (coo_array((Tq1, [q.ii, q.j]), (q.N, nb)).toarray() @ V)

    Tq2 = q.Gij * np.cos(U) + q.Bij * np.sin(U)
    Qi_Ti = Vb * (coo_array((Tq2, [q.ii, q.j]), (q.N, nb)).toarray() @ V) - (Vb ** 2 * q.Gii)
    Qi_Tj = -Vi[q.ij] * Vj[q.ij] * Tq2[q.ij]

    Qi_Vi = (coo_array((Tq1, [q.ii, q.j]), (q.N, nb)).toarray() @ V) - (Vb * q.Bii)
    Qi_Vj = Vi[q.ij] * Tq1[q.ij]

    J51 = coo_array((np.r_[Qi_Ti, Qi_Tj], [q.jci, q.jcj]), (q.N, nb)).toarray()
    J52 = coo_array((np.r_[Qi_Vi, Qi_Vj], [q.jci, q.jcj]), (q.N, nb)).toarray()

    Fi = np.r_[Pi, Qi]
    Ji = np.block([[J41, J42], [J51, J52]])

    return Fi, Ji


def flow_acse(V, T, p, q, nb):
    U = T[p.i] - T[p.j] - p.fij
    Vi = V[p.i]
    Vj = V[p.j]

    Tp1 = p.gij * np.cos(U) + p.bij * np.sin(U)
    Pij = Vi ** 2 * p.tgij - p.pij * Vi * Vj * Tp1

    Tp2 = p.gij * np.sin(U) - p.bij * np.cos(U)
    Pij_Ti = p.pij * Vi * Vj * Tp2
    Pij_Tj = -Pij_Ti

    Pij_Vi = 2 * p.tgij * Vi - p.pij * Vj * Tp1
    Pij_Vj = -p.pij * Vi * Tp1

    J11 = coo_array((np.r_[Pij_Ti, Pij_Tj], [p.jci, p.jcj]), (p.N, nb)).toarray()
    J12 = coo_array((np.r_[Pij_Vi, Pij_Vj], [p.jci, p.jcj]), (p.N, nb)).toarray()

    U = T[q.i] - T[q.j] - q.fij
    Vi = V[q.i]
    Vj = V[q.j]

    Tq1 = q.gij * np.sin(U) - q.bij * np.cos(U)
    Qij = -q.tbij * Vi ** 2 - q.pij * Vi * Vj * Tq1

    Tq2 = q.gij * np.cos(U) + q.bij * np.sin(U)
    Qij_Ti = -q.pij * Vi * Vj * Tq2
    Qij_Tj = -Qij_Ti

    Qij_Vi = -2 * q.tbij * Vi - q.pij * Vj * Tq1
    Qij_Vj = -q.pij * Vi * Tq1

    J21 = coo_array((np.r_[Qij_Ti, Qij_Tj], [q.jci, q.jcj]), (q.N, nb)).toarray()
    J22 = coo_array((np.r_[Qij_Vi, Qij_Vj], [q.jci, q.jcj]), (q.N, nb)).toarray()

    Ff = np.r_[Pij, Qij]
    Jf = np.block([[J11, J12], [J21, J22]])

    return Ff, Jf


def current_acse(V, T, c, nb):
    U = T[c.i] - T[c.j] - c.fij
    Vi = V[c.i]
    Vj = V[c.j]

    Tc1 = c.C * np.cos(U) - c.D * np.sin(U)
    Fc = np.sqrt(c.A * Vi ** 2 + c.B * Vj ** 2 - 2 * Vi * Vj * Tc1)
    mask = Fc != 0

    Tc2 = c.C * np.sin(U) + c.D * np.cos(U)
    Iij_Ti = np.zeros_like(Fc)
    Iij_Ti[mask] = (Vi[mask] * Vj[mask] * Tc2[mask]) / Fc[mask]
    Iij_Tj = -Iij_Ti

    Iij_Vi = np.zeros_like(Fc)
    Iij_Vj = np.zeros_like(Fc)
    Iij_Vi[mask] = (-Vj[mask] * Tc1[mask] + c.A[mask] * Vi[mask]) / Fc[mask]
    Iij_Vj[mask] = (-Vi[mask] * Tc1[mask] + c.B[mask] * Vj[mask]) / Fc[mask]

    J31 = coo_array((np.r_[Iij_Ti, Iij_Tj], [c.jci, c.jcj]), (c.N, nb)).toarray()
    J32 = coo_array((np.r_[Iij_Vi, Iij_Vj], [c.jci, c.jcj]), (c.N, nb)).toarray()

    Jc = np.c_[J31, J32]

    return Fc, Jc


def voltage_acse(V, vm, nb):
    Fv = V[vm.i]

    V_V = coo_array((np.ones(vm.N), [np.arange(vm.N), vm.i]), (vm.N, nb)).toarray()
    V_T = np.zeros((vm.N, nb))

    Jv = np.c_[V_T, V_V]

    return Fv, Jv

class H_AC:
    def __init__(self, sys, branch, meas_idx):
        Pi_idx = Qi_idx  = Vm_idx = np.zeros(sys.nb).astype(bool)
        Pf_idx = Qf_idx = Cm_idx = np.zeros_like(branch.i).astype(bool)
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
        Fc, Jc = current_acse(V, T,self.cm, self.nb)
        Fv, Jv = voltage_acse(V, self.vm, self.nb)

        f = np.r_[Ff, Fc, Fi, Fv]
        J = np.r_[Jf, Jc, Ji, Jv]

        return f, J
