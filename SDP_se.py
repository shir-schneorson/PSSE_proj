import numpy as np
import cvxpy as cp



def SDP_se(measurements, variance, slk_bus, h_ac, nb, num_random_samples=50):
    """
    Solves PSSE using SDP relaxation (epigraph trick + Schur complement).
    Outputs two estimates:
      - Leading eigenvector heuristic
      - Best random sample heuristic
    """

    L = len(measurements)

    # SDP variables
    V = cp.Variable((nb, nb), hermitian=True, name="V")
    X = cp.Variable(L, name="X")
    z = measurements
    w = variance

    H = h_ac.H
    # Constraints
    constraints = [V >> 0]  # V must be PSD
    for l in range(L):
        residual = z[l] - cp.trace(H[l] @ V)
        schur_matrix = cp.bmat([
            [cp.reshape(X[l], (1, 1) , 'C'), cp.reshape(residual, (1, 1), 'C')],
            [cp.reshape(residual, (1, 1), 'C'), np.ones((1, 1))]
        ])
        constraints.append(schur_matrix >> 0)


    # Objective
    objective = cp.Minimize(cp.matmul(w, X))

    # Solve
    prob = cp.Problem(objective, constraints)

    try:
        prob.solve(solver=cp.CVXOPT, verbose=False)
        Converged = True
    except cp.SolverError:
        Converged = False

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError(f"Solver status: {prob.status}")

    # Retrieve solution
    V_opt = V.value

    # Heuristic 1: Leading eigenvector
    eigvals, eigvecs = np.linalg.eigh(V_opt)
    idx_max = np.argmax(eigvals)
    v_best = np.sqrt(eigvals[idx_max]) * eigvecs[:, idx_max]


    return v_best, V_opt, Converged
