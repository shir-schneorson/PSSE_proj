# h_ac_torch.py
import torch
from SE_torch.net_preprocess.compose_measurement import Pi, Qi, Pf, Qf, Vm, Cm

# default to float64 for numerical stability; complex → complex128
torch.set_default_dtype(torch.float64)


def _to_c128(x, device=None):
    """Convert array-like (numpy/torch) to torch.complex128 on device."""
    t = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
    if t.is_complex():
        return t.to(device=device, dtype=torch.complex128)
    # real → complex
    return t.to(device=device, dtype=torch.float64).to(torch.complex128)


def _to_f64(x, device=None):
    """Convert array-like to torch.float64 on device."""
    return (x if isinstance(x, torch.Tensor) else torch.as_tensor(x)).to(device=device, dtype=torch.float64)


def Hinj(p, q, nb, device=None):
    I = torch.eye(nb, device=device, dtype=torch.float64)

    # ----- P injections -----
    IPi = I[p.idx]
    outer_eii_Pi = torch.einsum('ik,il->ikl', IPi, IPi)  # (Lp, nb, nb)
    YPi = torch.matmul(outer_eii_Pi.to(torch.complex128), p.Ybu)   # (Lp, nb, nb)
    HPi = 0.5 * (YPi + torch.conj(YPi.transpose(1, 2)))

    # ----- Q injections -----
    IQi = I[q.idx]
    outer_eii_Qi = torch.einsum('ik,il->ikl', IQi, IQi)
    YQi = torch.matmul(outer_eii_Qi.to(torch.complex128), q.Ybu)
    HQi = 0.5j * (YQi - torch.conj(YQi.transpose(1, 2)))

    return torch.cat([HPi, HQi], dim=0)  # (Lp+Lq, nb, nb)


def Hflow(pf, qf, nb, device=None):
    I = torch.eye(nb, device=device, dtype=torch.float64)

    # ----- P flows -----
    IPi = I[pf.i]
    IPj = I[pf.j]
    eii_P = torch.einsum('ik,il->ikl', IPi, IPi)  # (Lf, nb, nb)
    eij_P = torch.einsum('ik,il->ikl', IPi, IPj)

    ysi_p = pf.ysi.unsqueeze(1).unsqueeze(2)
    yij_p = pf.yij.unsqueeze(1).unsqueeze(2)
    YPij = eii_P.to(torch.complex128) * ysi_p + eij_P.to(torch.complex128) * yij_p
    HPij = 0.5 * (YPij + torch.conj(YPij.transpose(1, 2)))

    # ----- Q flows -----
    IQi = I[qf.i]
    IQj = I[qf.j]
    eii_Q = torch.einsum('ik,il->ikl', IQi, IQi)
    eij_Q = torch.einsum('ik,il->ikl', IQi, IQj)

    ysi_q = qf.ysi.unsqueeze(1).unsqueeze(2)
    yij_q = qf.yij.unsqueeze(1).unsqueeze(2)
    YQij = eii_Q.to(torch.complex128) * ysi_q + eij_Q.to(torch.complex128) * yij_q
    HQij = 0.5j * (YQij - torch.conj(YQij.transpose(1, 2)))

    return torch.cat([HPij, HQij], dim=0)


def Hcm(cm, nb, device=None):
    I = torch.eye(nb, device=device, dtype=torch.float64)

    Ii = I[cm.i]
    Ij = I[cm.j]

    eii = torch.einsum('ik,il->ikl', Ii, Ii)  # (L, nb, nb)
    eij = torch.einsum('ik,il->ikl', Ii, Ij)
    eji = torch.einsum('ik,il->ikl', Ij, Ii)
    ejj = torch.einsum('ik,il->ikl', Ij, Ij)

    ysi = cm.ysi.unsqueeze(1).unsqueeze(2)
    yij = cm.yij.unsqueeze(1).unsqueeze(2)

    Hij = (
        ysi * torch.conj(ysi) * eii.to(torch.complex128)
        + torch.conj(ysi) * yij * eij.to(torch.complex128)
        + ysi * torch.conj(yij) * eji.to(torch.complex128)
        + yij * torch.conj(yij) * ejj.to(torch.complex128)
    )
    return Hij  # (L, nb, nb)


def Hvm(vm, nb, device=None):
    I = torch.eye(nb, device=device, dtype=torch.float64)
    IVmi = I[vm.i]
    eii_V = torch.einsum('ik,il->ikl', IVmi, IVmi)  # (Lv, nb, nb)
    return eii_V.to(torch.complex128)  # keep complex dtype for uniformity


class H_AC:
    def __init__(self, sys, branch, meas_idx, device=None):
        # default masks
        Pi_idx = Qi_idx = Vm_idx = torch.zeros(sys.nb, dtype=torch.bool)
        Pf_idx = Qf_idx = Cm_idx = torch.zeros_like(torch.as_tensor(branch.i, dtype=torch.long), dtype=torch.bool)

        # accept external masks (numpy/torch/bool array)
        if meas_idx.get('Pi_idx') is not None: Pi_idx = torch.as_tensor(meas_idx['Pi_idx'], dtype=torch.bool)
        if meas_idx.get('Qi_idx') is not None: Qi_idx = torch.as_tensor(meas_idx['Qi_idx'], dtype=torch.bool)
        if meas_idx.get('Pf_idx') is not None: Pf_idx = torch.as_tensor(meas_idx['Pf_idx'], dtype=torch.bool)
        if meas_idx.get('Qf_idx') is not None: Qf_idx = torch.as_tensor(meas_idx['Qf_idx'], dtype=torch.bool)
        if meas_idx.get('Cm_idx') is not None: Cm_idx = torch.as_tensor(meas_idx['Cm_idx'], dtype=torch.bool)
        if meas_idx.get('Vm_idx') is not None: Vm_idx = torch.as_tensor(meas_idx['Vm_idx'], dtype=torch.bool)

        # build measurement objects (they may hold numpy; we'll convert inside H* funcs)
        self.pi = Pi(Pi_idx, sys.bus, sys.Ybus, sys.Yij, sys.Yii)
        self.qi = Qi(Qi_idx, sys.bus, sys.Ybus, sys.Yij, sys.Yii)
        self.pf = Pf(Pf_idx, branch)
        self.qf = Qf(Qf_idx, branch)
        self.cm = Cm(Cm_idx, sys.bus, branch)
        self.vm = Vm(Vm_idx, sys.bus)

        self.nb = sys.nb
        self.device = device

        # assemble H (complex128)
        Hf = Hflow(self.pf, self.qf, self.nb, device=self.device)
        Hc = Hcm(self.cm, self.nb, device=self.device)
        Hi = Hinj(self.pi, self.qi, self.nb, device=self.device)
        Hv = Hvm(self.vm, self.nb, device=self.device)
        self.H = torch.cat([Hf, Hc, Hi, Hv], dim=0)  # (M, nb, nb) complex128

    @torch.no_grad()
    def estimate(self, Vc: torch.Tensor):
        """
        Vc: complex bus-voltage vector (nb,) as torch.complex128
        Returns:
          z_est: (M,) real — each is trace(H_k @ (v v^H))
          J_feat: (M, 2*nb) real — concatenation of angle(H_k v) and abs(H_k v)
        """
        # Ensure complex on same device
        Vc = _to_c128(Vc, device=self.H.device)

        # V = v v^H
        V = torch.outer(Vc, torch.conj(Vc))  # (nb, nb) complex128

        # z_k = trace(H_k @ V)
        HV = torch.matmul(self.H, V)                    # (M, nb, nb)
        z_est = HV.diagonal(dim1=1, dim2=2).sum(-1).real  # (M,)

        # J_k = - H_k v
        J = -torch.matmul(self.H, Vc.view(-1, 1)).view(self.H.size(0), -1)  # (M, nb) complex
        J_feat = torch.cat([torch.angle(J), torch.abs(J)], dim=1)           # (M, 2*nb) real

        return z_est, J_feat