import numpy as np

from init_net.compose_meausrement import Pi, Qi, Pf, Qf, Vm, Cm


def Hinj(p, q, nb):
    I = np.eye(nb)

    IPi = I[p.idx]
    outer_eii_Pi =  np.einsum('ik,il->ikl', IPi, IPi)
    YPi = outer_eii_Pi @ p.Ybu
    HPi = .5 * (YPi + np.conj(YPi.transpose((0, 2, 1))))

    IQi = I[q.idx]
    outer_eii_Qi = np.einsum('ik,il->ikl', IQi, IQi)
    YQi = outer_eii_Qi @ q.Ybu
    HQi = .5j * (YQi - np.conj(YQi.transpose((0, 2, 1))))
    return np.r_[HPi, HQi]


def Hflow(pf, qf, nb):
    I = np.eye(nb)

    IPi = I[pf.i]
    IPj = I[pf.j]
    outer_eii_Pij =  np.einsum('ik,il->ikl', IPi, IPi)
    outer_eij_Pij = np.einsum('ik,il->ikl', IPi, IPj)
    YPij = outer_eii_Pij * pf.ysi[:, np.newaxis, np.newaxis] + outer_eij_Pij * pf.yij[:, np.newaxis, np.newaxis]
    HPij = .5 * (YPij + np.conj(YPij.transpose((0, 2, 1))))

    IQi = I[qf.i]
    IQj = I[qf.j]
    outer_eii_Qij =  np.einsum('ik,il->ikl', IQi, IQi)
    outer_eij_Qij = np.einsum('ik,il->ikl', IQi, IQj)
    YQij = outer_eii_Qij * qf.ysi[:, np.newaxis, np.newaxis] + outer_eij_Qij * qf.yij[:, np.newaxis, np.newaxis]
    HQij = .5j * (YQij - np.conj(YQij.transpose((0, 2, 1))))
    return np.r_[HPij, HQij]


def Hcm(cm, nb):
    I = np.eye(nb)

    # Get canonical basis vectors for buses i and j
    Ii = I[cm.i]  # shape: (L, nb)
    Ij = I[cm.j]  # shape: (L, nb)

    # Outer products to get projection matrices
    eii = np.einsum('ik,il->ikl', Ii, Ii)  # shape: (L, nb, nb)
    eij = np.einsum('ik,il->ikl', Ii, Ij)
    eji = np.einsum('ik,il->ikl', Ij, Ii)
    ejj = np.einsum('ik,il->ikl', Ij, Ij)

    # Build Hermitian matrices H_ij such that:
    #     vᴴ @ H_ij @ v = |I_ij|²

    # Expand admittances to broadcast properly
    ysi = cm.ysi[:, np.newaxis, np.newaxis]
    yij = cm.yij[:, np.newaxis, np.newaxis]

    # Construct H_ij
    Hij = (
            ysi * ysi.conj() * eii
            + ysi.conj() * yij * eij
            + ysi * yij.conj() * eji
            + yij * yij.conj() * ejj
    )
    return Hij



def Hvm(vm, nb):
    I = np.eye(nb)
    IVmi = I[vm.i]
    outer_eii_Vm =  np.einsum('ik,il->ikl', IVmi, IVmi)

    return outer_eii_Vm


class H_AC:
    def __init__(self,  sys, branch, meas_idx):
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

        Hf = Hflow(self.pf, self.qf, self.nb)
        Hc = Hcm(self.cm, self.nb)
        Hi = Hinj(self.pi, self.qi, self.nb)
        Hv = Hvm(self.vm, self.nb)
        self.H = np.r_[Hf, Hc, Hi, Hv]


    def estimate(self, Vc):
        V = np.outer(Vc, np.conj(Vc))
        z_est = np.linalg.trace(np.einsum('ijk,kl', self.H, V))
        J = - np.einsum('ijk,k->ij', self.H, Vc)
        J = np.c_[np.angle(J), np.abs(J)]
        return z_est, J

