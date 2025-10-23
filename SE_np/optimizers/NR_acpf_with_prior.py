import numpy as np
import time  # you referenced time.*; keep this import

from SE_np.optimizers.NR_acpf import qv_limits, idx_par1, idx_par2, data_jacobian, jacobian11, jacobian12, jacobian21, \
    jacobian22, cq, cv


def NR_PF(sys, loads, gens, x_init, user, m=None, Q=None, R=None, reg_scale=1.0, jitter=1e-10):
    """
    Newton–Raphson AC power flow with optional Gaussian prior on the FULL state x=[T;V].

    Args
    ----
    sys: object with fields nb (number of buses), slk_bus (array-like, first is slack angle index), Ybus, etc.
    loads: (nb, 2) array with columns [Pl, Ql]
    gens:  (nb, 2) array with columns [Pg, Qg]
    x_init: (nb, 2) initial state [T, V] per-bus
    user: dict with keys: 'stop', 'maxIter', 'list' (e.g., {'reactive', 'voltage', ...})
    m, Q:  Optional prior mean and covariance for the FULL state vector [T(0:nb), V(0:nb)].
           Shapes: m: (2*nb,), Q: (2*nb, 2*nb). If provided, used as MAP regularization.
    reg_scale: scalar to scale the prior precision (Q^{-1}) strength.
    jitter: small diagonal added to the reduced precision for numerical stability.

    Returns
    -------
    pf: dict with solution, timings, iterations; pf['Vc'] is final complex voltages.
    """
    pf = {}
    pf['method'] = 'AC Power Flow using Newton-Raphson Algorithm'
    pf['limit'] = np.zeros((sys.nb, 2))

    T = x_init[:, 0].astype(float).copy()
    V = x_init[:, 1].astype(float).copy()
    w = np.diag(R)

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

    # ---------- Prior setup (reduce from FULL x=[T_all; V_all] to solver x=[T_wo_slack; V_pq]) ----------
    use_prior = (m is not None) and (Q is not None)

    pf['time'] = {}
    start_pre = time.time()
    pf['time']['pre'] = time.time() - start_pre
    start_con = time.time()

    while np.max(np.abs(DelPQ)) > user['stop'] and No < user['maxIter']:
        No += 1

        if 'reactive' in user['list']:
            sys, alg, idx, pf, Qcon, V, T, Qg, Qgl, DelPQ = cq(
                sys, alg, idx, pf, Qcon, V, T, Qg, Ql, Qgl, Pgl, DelPQ
            )

        if 'voltage' in user['list']:
            sys, alg, idx, pf, Vcon, DelPQ, V, T = cv(
                sys, alg, idx, pf, Vcon, DelPQ, V, T, Pgl, Qgl
            )

        # Jacobian blocks and assembly (square J matching reduced variables)
        alg = data_jacobian(T, V, alg, sys.nb)
        J11 = jacobian11(V, alg, idx['j11'], sys.nb)
        J12 = jacobian12(V, alg, idx['j12'], sys.nb)
        J21 = jacobian21(alg, idx['j21'], sys.nb)
        J22 = jacobian22(V, alg, idx['j22'])
        J = np.block([[J11, J12],
                      [J21, J22]])  # shape (M, N) with N = (nb-1) + |pq|

        # --- Compute NR/MAP step ---
        if use_prior:
            red_full_idx = np.concatenate([alg['ii'], alg['pq'] + sys.nb])
            x = np.concatenate((T, V))
            Q_red = reg_scale * Q[np.ix_(red_full_idx, red_full_idx)]

            # Build reduced precision Qinv_red = reg_scale * (Q_red^{-1}) with a robust factorization
            # We do NOT explicitly keep L; we’ll use Qinv_red directly.
            # R_red = R[np.ix_(red_full_idx, red_full_idx)]

            DelPrior = (x - m)[red_full_idx]

            # JT_R_J = J.T @ R_red @ J
            # rhs = -J.T @ R_red @ DelPQ
            # rhs -= Q_red @ DelPrior
            # A = JT_R_J + Q_red
            #
            # dx = np.linalg.solve(A, rhs)

            w_red = w[red_full_idx]  # or keep as a 1D vector from the start
            JW = J * w_red[:, None]  # row-scale J by w
            A = J.T @ JW + Q_red
            rhs = -(J.T @ (w_red * DelPQ)) - Q_red @ DelPrior

            L = np.linalg.cholesky(A)
            dx = np.linalg.solve(L.T, np.linalg.solve(L, rhs))
        else:
            dx = -np.linalg.solve(J, DelPQ)

        dx = np.insert(dx, sys.slk_bus[0], 0.0)
        dTV = dx

        # Update state in the solver’s ordering [T(all), V(pq)]
        TV = np.concatenate((T, V[alg['pq']])) + dTV
        T = TV[:sys.nb]
        V[alg['pq']] = TV[sys.nb:]

        # Residuals for next iteration
        Vc = V * np.exp(1j * T)
        DelS = Vc * np.conj(sys.Ybus @ Vc) - (Pgl + 1j * Qgl)
        DelPQ = np.concatenate((np.real(DelS[alg['ii']]), np.imag(DelS[alg['pq']])))


    pf['Vc'] = Vc
    pf['time']['con'] = time.time() - start_con
    pf['iteration'] = No

    return pf