import numpy as np

import numpy as np
from scipy.sparse import coo_array


def Hinj(p, q, nb):
    I = np.eye(nb)

    IPi = I[p.idx]
    outer_eii_Pi =  np.einsum('ik,il->ikl', IPi, np.conj(IPi))
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



def Hvm(vm, nb):
    I = np.eye(nb)
    IVmi = I[vm.i]
    outer_eii_Vm =  np.einsum('ik,il->ikl', IVmi, IVmi)

    return outer_eii_Vm


class H_AC:
    def __init__(self, pi, qi, pf, qf, vm, nb):
        self.pi = pi
        self.qi = qi
        self.pf = pf
        self.qf = qf
        self.vm = vm
        self.nb = nb
        Hf = Hflow(self.pf, self.qf, self.nb)
        Hi = Hinj(self.pi, self.qi, self.nb)
        Hv = Hvm(self.vm, self.nb)
        self.H = np.r_[Hf, Hi, Hv]


    def estimate(self, Vc):
        V = np.outer(Vc, np.conj(Vc))
        z_est = np.linalg.trace(np.einsum('ijk,kl', self.H, V))
        return z_est
