import numpy as np
import cvxpy as cp



def SDP_se(measurements, variance, slk_bus, h_ac, nb, x0, num_random_samples=1000):
    """
    Solves PSSE using SDP relaxation (epigraph trick + Schur complement).
    Outputs two estimates:
      - Leading eigenvector heuristic
      - Best random sample heuristic
    """

    T = x0[:nb]
    T[slk_bus[0]] = slk_bus[1]
    Vm = x0[nb:]
    Vc = Vm * np.cos(T) + 1j * Vm * np.sin(T)
    V0 = np.outer(Vc, np.conj(Vc))

    L = len(measurements)

    # SDP variables
    V = cp.Variable((nb, nb), complex=True, name="V")
    # V.value = V0
    X = cp.Variable(L, name="X")
    z = cp.Parameter(L, value=measurements, name="z")
    w = cp.Parameter(L, value=variance, name="w")

    H = h_ac.H
    # Constraints
    constraints = [V >> 0]  # V must be PSD
    for l in range(L):
        residual = z[l] - cp.trace(H[l] @ V)
        schur_matrix = cp.bmat([
            [cp.reshape(-X[l], (1, 1)), cp.reshape(residual, (1, 1))],
            [cp.reshape(residual, (1, 1)), cp.Constant([[-1]])]
        ])
        constraints.append(schur_matrix << 0)


    # Objective
    objective = cp.Minimize(w @ X)

    # Solve
    prob = cp.Problem(objective, constraints,)
    prob.solve(verbose=True)

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError(f"Solver status: {prob.status}")

    # Retrieve solution
    V_opt = V.value

    # Heuristic 1: Leading eigenvector
    eigvals, eigvecs = np.linalg.eigh(V_opt)
    idx_max = np.argmax(eigvals)
    v_leading = np.sqrt(eigvals[idx_max]) * eigvecs[:, idx_max]

    # Heuristic 2: Random samples
    best_fit = np.inf
    v_best = None

    for _ in range(num_random_samples):
        v_rand = (np.random.randn(nb) + 1j * np.random.randn(nb)) / np.sqrt(2)

        # Step 2: apply covariance
        L = np.linalg.cholesky(V_opt+ np.eye(V_opt.shape[0]) * 1e-4)
        v_rand = L @ v_rand

        # Evaluate fit: sum of squared residuals
        residuals = z - h_ac.estimate(v_rand)

        fit =( cp.norm(residuals) ** 2).value

        if fit < best_fit:
            best_fit = fit
            v_best = v_rand

    return v_leading, v_best
