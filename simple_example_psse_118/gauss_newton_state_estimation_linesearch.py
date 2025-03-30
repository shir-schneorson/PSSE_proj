import numpy as np

from compute_jaccobian import compute_jacobian
from compute_power_flow import compute_power_flow
from H_AC_matrix import H_AC_matrix


def gauss_newton_state_estimation_linesearch(P_meas, Q_meas, Ybus, slack_bus, tol, max_iter):
    """
    Gauss-Newton method for AC state estimation with backtracking line search.

    Parameters:
    P_meas : ndarray
        Active power injections (n,).
    Q_meas : ndarray
        Reactive power injections (n,).
    Ybus : ndarray
        Bus admittance matrix (n x n).
    slack_bus : int
        Index of slack bus (zero-based index).
    tol : float
        Convergence tolerance (e.g., 1e-6).
    max_iter : int
        Maximum number of iterations.

    Returns:
    theta_est : ndarray
        Estimated voltage angles (n,).
    V_est : ndarray
        Estimated voltage magnitudes (n,).
    """
    n = len(Ybus)

    # Initial guess: Flat start (V=1 p.u., θ=0)
    theta = np.zeros(n)
    V = np.ones(n)

    # Fix slack bus values
    theta[slack_bus] = 0
    V[slack_bus] = 1

    # Extract Ybus components
    G = np.real(Ybus)
    B = np.imag(Ybus)

    # Remove slack bus from Jacobian
    non_slack = np.setdiff1d(np.arange(n), slack_bus)

    # Iterative Gauss-Newton updates
    for _ in range(max_iter):
        # Compute power injections
        P_est, Q_est = compute_power_flow(theta, V, G, B)
        # P_est, Q_est = H_AC_matrix(np.concatenate([theta, V]), Ybus)

        # Compute residuals
        delta_P = P_meas - P_est
        delta_Q = Q_meas - Q_est
        delta_y = np.concatenate((delta_P[non_slack], delta_Q[non_slack]))

        # Compute Jacobian
        J = compute_jacobian(theta, V, G, B, non_slack)

        # Solve for state update using least squares
        delta_x = np.linalg.lstsq(J.T @ J, J.T @ delta_y, rcond=None)[0]
        # delta_x = np.linalg.pinv(J) @ delta_y

        # **Backtracking Line Search for Step Size**
        alpha = 1.0  # Initial step size
        beta = 0.5  # Step size reduction factor
        c = 0.1  # Sufficient decrease condition factor
        max_ls_iter = 10  # Max iterations for line search

        # Compute initial cost function (mismatch norm)
        norm_f_old = np.linalg.norm(delta_y)

        # Try reducing step size until it improves mismatch
        for _ in range(max_ls_iter):
            # Compute tentative updates
            theta_temp = theta.copy()
            V_temp = V.copy()
            theta_temp[non_slack] += alpha * delta_x[:len(non_slack)]
            V_temp[non_slack] += alpha * delta_x[len(non_slack):]

            # Compute power injections for new state
            P_new, Q_new = compute_power_flow(theta_temp, V_temp, G, B)
            # P_new, Q_new = H_AC_matrix(np.concatenate([theta_temp, V_temp]), Ybus)
            delta_P_new = P_meas - P_new
            delta_Q_new = Q_meas - Q_new
            delta_y_new = np.concatenate((delta_P_new[non_slack], delta_Q_new[non_slack]))

            # Compute new cost function (mismatch norm)
            norm_f_new = np.linalg.norm(delta_y_new)

            # Armijo condition: Check if the step improves the mismatch
            if norm_f_new < (1 - c * alpha) * norm_f_old:
                break  # Accept the step size
            else:
                alpha *= beta  # Reduce step size

        # Apply update with optimal step size
        theta[non_slack] += alpha * delta_x[:len(non_slack)]
        V[non_slack] += alpha * delta_x[len(non_slack):]

        # Check convergence
        if np.linalg.norm(delta_x) < tol:
            print('converged')
            break

    return theta, V
