import time

import numpy as np
from scipy.sparse import coo_matrix


def qv_limits(user, sys, Ql):
    on = np.zeros((sys.nb, 1))
    Qcon, Vcon = np.zeros((sys.nb, 3)), np.zeros((sys.nb, 3))

    if 'reactive' in user['list']:
        pv_mask = (sys.bus.bus_type.values == 2) & (sys.bus.Qmin.values < sys.bus.Qmax.values)
        on[pv_mask] = 1

        q_min = sys.bus.Qmin.values.reshape(-1, 1)
        q_max = sys.bus.Qmax.values.reshape(-1, 1)
        q = Ql.reshape(-1, 1)
        Qcon = np.hstack((q_min - q, q_max - q, on))

    elif 'voltage' in user['list']:
        pq_mask = (sys.bus.bus_type.values == 1) & (sys.bus.Vmin.values < sys.bus.Vmax.values)
        on[pq_mask] = 1

        v_min_v_max = sys.bus.loc[:, ['Vmin', 'Vmax']].values
        Vcon = np.hstack((v_min_v_max, on))

    return Qcon, Vcon


def idx_par1(sys):
    Yte = sys.Yij.copy()

    Yij_mod = sys.Yij.copy()
    Yij_mod[sys.slk_bus[0], :] = 0

    alg = {}
    idx = {}

    alg['i'], alg['j'] = np.nonzero(Yij_mod)

    GijBij = sys.Ybus[alg['i'], alg['j']]
    alg['Gij'] = np.real(GijBij)
    alg['Bij'] = np.imag(GijBij)

    Yte = Yte.copy()
    Yte = np.delete(Yte, sys.slk_bus[0], axis=0)
    alg['fd1i'], _ = np.nonzero(Yte)

    alg['ii'] = sys.bus.idx_bus.values.astype(int).copy()
    alg['ii'] = np.delete(alg['ii'], sys.slk_bus[0])

    idx['j11'] = {}
    idx['j11']['ij'] = alg['j'] != (sys.slk_bus[0])

    Yte = np.delete(Yte, sys.slk_bus[0], axis=1)
    q, w = np.nonzero(Yte)

    bu = np.arange(sys.nb - 1)
    idx['j11']['jci'] = np.concatenate((bu, q))
    idx['j11']['jcj'] = np.concatenate((bu, w))

    return alg, idx


def idx_par2(sys, alg, idx):
    alg['pq'] = np.where(sys.bus.bus_type.values == 1)[0]
    alg['Npq'] = len(alg['pq'])

    alg['fdi'], alg['fdj'] = np.nonzero(sys.Yij[alg['pq'], :])
    alg['fij'] = np.isin(alg['i'], alg['pq'])

    pq_idx = alg['pq']
    GiiBii = sys.Ybus[pq_idx, pq_idx]
    alg['Gii'] = np.real(GiiBii)
    alg['Bii'] = np.imag(GiiBii)

    idx['j12'] = {}
    idx['j12']['ij'] = np.isin(alg['j'], alg['pq'])
    idx['j12']['i'] = alg['i'][idx['j12']['ij']]

    Yii = sys.Yii[:, pq_idx].copy()
    Yii = np.delete(Yii, sys.slk_bus[0], axis=0)
    c, d = np.nonzero(Yii)

    Yij = sys.Yij[:, pq_idx].copy()
    Yij = np.delete(Yij, sys.slk_bus[0], axis=0)
    q, w = np.nonzero(Yij)

    idx['j12']['jci'] = np.concatenate([c, q])
    idx['j12']['jcj'] = np.concatenate([d, w])

    idx['j21'] = {}
    mn = np.isin(alg['i'], alg['pq'])
    idx['j21']['ij'] = np.logical_and(idx['j11']['ij'], mn)

    Yii = sys.Yii[pq_idx, :].copy()
    Yii = np.delete(Yii, sys.slk_bus[0], axis=1)
    c, d = np.nonzero(Yii)

    Yij = sys.Yij[pq_idx, :].copy()
    Yij = np.delete(Yij, sys.slk_bus[0], axis=1)
    q, w = np.nonzero(Yij)

    if alg['Npq'] == 1:
        idx['j21']['jci'] = np.concatenate([c, q.ravel()])
        idx['j21']['jcj'] = np.concatenate([d, w.ravel()])
    else:
        idx['j21']['jci'] = np.concatenate([c, q])
        idx['j21']['jcj'] = np.concatenate([d, w])

    idx['j22'] = {}
    idx['j22']['ij'] = np.logical_and(idx['j12']['ij'], mn)
    idx['j22']['i'] = alg['i'][idx['j22']['ij']]

    Yii_pq = sys.Yii[pq_idx, :][:, pq_idx]
    c, d = np.nonzero(Yii_pq)

    Yij_pq = sys.Yij[pq_idx, :][:, pq_idx]
    q, w = np.nonzero(Yij_pq)

    idx['j22']['jci'] = np.concatenate([c, q])
    idx['j22']['jcj'] = np.concatenate([d, w])

    return alg, idx


def cq(sys, alg, idx, pf, Qcon, V, T, Qg, Ql, Qgl, Pgl, DelPQ):
    Vc = V * np.exp(1j * T)
    S = Vc * np.conj(sys.Ybus @ Vc)
    Q = -np.imag(S)

    Qmin_violated = np.where((Q < Qcon[:, 0]) & (Qcon[:, 2] == 1))[0]
    Qmax_violated = np.where((Q > Qcon[:, 1]) & (Qcon[:, 2] == 1))[0]

    if Qmin_violated.size > 0:
        sys.bus.loc[Qmin_violated, 'bus_type'] = 1
        Qcon[Qmin_violated, 2] = 0

        Qg[Qmin_violated] = Qcon[Qmin_violated, 0] + Ql[Qmin_violated]

        Y_diag = sys.Ybus[Qmin_violated, Qmin_violated]
        conj_Vc = np.conj(Vc[Qmin_violated])
        rhs = (Pgl[Qmin_violated] - 1j * Qcon[Qmin_violated, 0]) / conj_Vc
        Vc[Qmin_violated] = (1.0 / Y_diag) * (rhs - sys.Yij[Qmin_violated, :] @ Vc)

    if Qmax_violated.size > 0:
        sys.bus.loc[Qmax_violated, 'bus_type'] = 1
        Qcon[Qmax_violated, 2] = 0

        Qg[Qmax_violated] = Qcon[Qmax_violated, 1] + Ql[Qmax_violated]

        Y_diag = sys.Ybus[Qmax_violated, Qmax_violated]
        conj_Vc = np.conj(Vc[Qmax_violated])
        rhs = (Pgl[Qmax_violated] - 1j * sys.Qcon[Qmax_violated, 1]) / conj_Vc
        Vc[Qmax_violated] = (1.0 / Y_diag) * (rhs - sys.Yij[Qmax_violated, :] @ Vc)

    if Qmin_violated.size > 0 or Qmax_violated.size > 0:
        alg, idx = idx_par2(sys, alg, idx)

        T = np.angle(Vc)
        V = np.abs(Vc)
        Qgl = Qg - Ql

        DelS = Vc * np.conj(sys.Ybus @ Vc) - (Pgl + 1j * Qgl)
        DelPQ = np.concatenate([np.real(DelS[alg['ii']]), np.imag(DelS[alg['pq']])])

        pf['limit'][Qmin_violated, 0] = 1
        pf['limit'][Qmax_violated, 1] = 1

    return sys, alg, idx, pf, Qcon, V, T, Qg, Qgl, DelPQ


def cv(sys, alg, idx, pf, Vcon, DelPQ, V, T, Pgl, Qgl):
    Vmin_violated = np.where((V < Vcon[:, 0]) & (Vcon[:, 2] == 1))[0]
    Vmax_violated = np.where((V > Vcon[:, 1]) & (Vcon[:, 2] == 1))[0]

    if Vmin_violated.size > 0:
        sys.bus.loc[Vmin_violated, 'bus_type'] = 2
        Vcon[Vmin_violated, 2] = 0

        V[Vmin_violated] = Vcon[Vmin_violated, 0]
        Vp = V * np.exp(1j * T)

        Q = -np.imag(np.conj(Vp[Vmin_violated]) * (sys.Ybus[Vmin_violated, :] @ Vp))

        Y_diag = sys.Ybus[Vmin_violated, Vmin_violated]
        rhs = (Pgl[Vmin_violated] - 1j * Q) / np.conj(Vp[Vmin_violated])
        T[Vmin_violated] = np.angle((1.0 / Y_diag) * (rhs - sys.Yij[Vmin_violated, :] @ Vp))

    if Vmax_violated.size > 0:
        sys.bus.loc[Vmax_violated, 'bus_type'] = 2
        Vcon[Vmax_violated, 2] = 0

        V[Vmax_violated] = Vcon[Vmax_violated, 1]
        Vp = V * np.exp(1j * T)

        Q = -np.imag(np.conj(Vp[Vmax_violated]) * (sys.Ybus[Vmax_violated, :] @ Vp))

        Y_diag = sys.Ybus[Vmax_violated, Vmax_violated]
        rhs = (Pgl[Vmax_violated] - 1j * Q) / np.conj(Vp[Vmax_violated])
        T[Vmax_violated] = np.angle((1.0 / Y_diag) * (rhs - sys.Yij[Vmax_violated, :] @ Vp))

    if Vmin_violated.size > 0 or Vmax_violated.size > 0:
        alg, idx = idx_par2(sys, alg, idx)

        Vp = V * np.exp(1j * T)
        DelS = Vp * np.conj(sys.Ybus @ Vp) - (Pgl + 1j * Qgl)
        DelPQ = np.concatenate([np.real(DelS[alg['ii']]), np.imag(DelS[alg['pq']])])

        pf['limit'][Vmin_violated, 0] = 1
        pf['limit'][Vmax_violated, 1] = 1

    return sys, alg, idx, pf, Vcon, DelPQ, V, T


def data_jacobian(T, V, alg, Nbu):
    Tij = T[alg['i']] - T[alg['j']]

    Te1 = (alg['Gij'] * np.sin(Tij)) - (alg['Bij'] * np.cos(Tij))
    Te2 = (alg['Gij'] * np.cos(Tij)) + (alg['Bij'] * np.sin(Tij))

    alg['Te1'] = Te1
    alg['Te2'] = Te2

    fD1 = coo_matrix(
        (-Te1, (alg['fd1i'], alg['j'])),
        shape=(Nbu - 1, Nbu)
    ).dot(V)
    alg['fD1'] = fD1

    Te2_pq = Te2[alg['fij']]
    fD2 = coo_matrix(
        (Te2_pq, (alg['fdi'], alg['fdj'])),
        shape=(alg['Npq'], Nbu)
    ).dot(V)
    alg['fD2'] = fD2

    Te1_pq = Te1[alg['fij']]
    fD3 = coo_matrix(
        (Te1_pq, (alg['fdi'], alg['fdj'])),
        shape=(alg['Npq'], Nbu)
    ).dot(V)
    alg['fD3'] = fD3

    alg['Vpq'] = V[alg['pq']]
    alg['Vij'] = V[alg['i']] * V[alg['j']]

    return alg


def jacobian11(V, alg, idx, Nbu):
    D = V[alg['ii']] * alg['fD1']  # Elementwise product

    N = alg['Vij'][idx['ij']] * alg['Te1'][idx['ij']]

    data = np.concatenate([D, N])
    row = idx['jci']
    col = idx['jcj']
    J11 = coo_matrix((data, (row, col)), shape=(Nbu - 1, Nbu - 1)).toarray()

    return J11


def jacobian12(V, alg, idx, Nbu):
    D = alg['fD2'] + 2 * alg['Gii'] * alg['Vpq']

    N = V[idx['i']] * alg['Te2'][idx['ij']]

    data = np.concatenate([D, N])
    row = idx['jci']
    col = idx['jcj']
    J12 = coo_matrix((data, (row, col)), shape=(Nbu - 1, alg['Npq'])).toarray()

    return J12


def jacobian21(alg, idx, Nbu):
    D = alg['Vpq'] * alg['fD2']  # Elementwise product

    N = -alg['Vij'][idx['ij']] * alg['Te2'][idx['ij']]

    data = np.concatenate([D, N])
    row = idx['jci']
    col = idx['jcj']
    J21 = coo_matrix((data, (row, col)), shape=(alg['Npq'], Nbu - 1)).toarray()

    return J21


def jacobian22(V, alg, idx):
    D = alg['fD3'] - 2 * alg['Bii'] * alg['Vpq']

    N = V[idx['i']] * alg['Te1'][idx['ij']]

    data = np.concatenate([D, N])
    row = idx['jci']
    col = idx['jcj']
    J22 = coo_matrix((data, (row, col)), shape=(alg['Npq'], alg['Npq'])).toarray()

    return J22


def NR_PF(sys, loads, gens, x_init, user):
    pf = {}
    pf['method'] = 'AC Power Flow using Newton-Raphson Algorithm'
    pf['limit'] = np.zeros((sys.nb, 2))

    T = x_init[:, 0]
    V = x_init[:, 1]

    Pl, Ql = loads[:, 0], loads[:, 1]
    Pg, Qg = gens[:, 0], gens[:, 1]

    No = 0

    Qcon, Vcon = qv_limits(user, sys, Ql)
    alg, idx = idx_par1(sys)
    alg, idx = idx_par2(sys, alg, idx)

    Vini = V * np.exp(1j * T)
    Pgl = Pg - Pl
    Qgl = Qg - Ql

    DelS = Vini * np.conj(sys.Ybus @ Vini) - (Pgl + 1j * Qgl)
    DelPQ = np.concatenate((np.real(DelS[alg['ii']]), np.imag(DelS[alg['pq']])))
    Vc = Vini.copy()

    pf['time'] = {}
    start_pre = time.time()
    pf['time']['pre'] = time.time() - start_pre
    start_con = time.time()

    while np.max(np.abs(DelPQ)) > user['stop'] and No < user['maxIter']:
        No += 1

        if 'reactive' in user['list']:
            sys, alg, idx, pf, Qcon, V, T, Qg, Qgl, DelPQ = cq(sys, alg, idx, pf, Qcon, V, T, Qg, Ql, Qgl, Pgl, DelPQ)

        if 'voltage' in user['list']:
            sys, alg, idx, pf, Vcon, DelPQ, V, T = cv(sys, alg, idx, pf, Vcon, DelPQ, V, T, Pgl, Qgl)

        alg = data_jacobian(T, V, alg, sys.nb)
        J11 = jacobian11(V, alg, idx['j11'], sys.nb)
        J12 = jacobian12(V, alg, idx['j12'], sys.nb)
        J21 = jacobian21(alg, idx['j21'], sys.nb)
        J22 = jacobian22(V, alg, idx['j22'])

        J = np.block([[J11, J12],
                      [J21, J22]])

        dTV = -np.linalg.solve(J, DelPQ)

        dTV = np.insert(dTV, sys.slk_bus[0], 0.0)

        TV = np.concatenate((T, V[alg['pq']])) + dTV

        T = TV[:sys.nb]
        V[alg['pq']] = TV[sys.nb:]

        Vc = V * np.exp(1j * T)
        DelS = Vc * np.conj(sys.Ybus @ Vc) - (Pgl + 1j * Qgl)
        DelPQ = np.concatenate((np.real(DelS[alg['ii']]), np.imag(DelS[alg['pq']])))

    pf['Vc'] = Vc
    pf['time']['con'] = time.time() - start_con
    pf['iteration'] = No

    return pf
