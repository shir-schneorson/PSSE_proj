import numpy as np

from compute_jaccobian import compute_jacobian
from compute_power_flow import compute_power_flow


def gauss_newton_state_estimation_real(P_meas, Q_meas, Ybus, slack_bus, tol, max_iter):
    """
    Gauss-Newton method for AC state estimation.

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

        # Compute residuals
        delta_P = P_meas - P_est
        delta_Q = Q_meas - Q_est
        delta_y = np.concatenate((delta_P[non_slack], delta_Q[non_slack]))

        # Compute Jacobian
        J = compute_jacobian(theta, V, G, B, non_slack)

        # Solve for state update using least squares
        delta_x = np.linalg.lstsq(J.T @ J, J.T @ delta_y, rcond=None)[0]

        # Update state variables with step size factors
        step_size1 = 0.225
        step_size2 = 0.225
        theta[non_slack] += step_size1 * delta_x[:len(non_slack)]
        V[non_slack] += step_size2 * delta_x[len(non_slack):]

        # Check convergence
        if np.linalg.norm(delta_x) < tol:
            break

    return theta, V
