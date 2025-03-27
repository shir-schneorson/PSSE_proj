import numpy as np


def compute_jacobian(theta, V, G, B, non_slack):
    """
    Computes the Jacobian matrix (excluding slack bus).

    Parameters:
    theta : ndarray
        Voltage angles (n,).
    V : ndarray
        Voltage magnitudes (n,).
    G : ndarray
        Conductance matrix (n x n).
    B : ndarray
        Susceptance matrix (n x n).
    non_slack : ndarray
        Indices of non-slack buses.

    Returns:
    J : ndarray
        Jacobian matrix.
    """
    n = len(V)

    # Compute angle differences
    Theta_diff = theta[:, np.newaxis] - theta[np.newaxis, :]

    # Compute Jacobian blocks
    H = -B * (V[:, np.newaxis] * V[np.newaxis, :]) * np.cos(Theta_diff) + \
        G * (V[:, np.newaxis] * V[np.newaxis, :]) * np.sin(Theta_diff)
    N = G * (V[:, np.newaxis] * V[np.newaxis, :]) * np.cos(Theta_diff) + \
        B * (V[:, np.newaxis] * V[np.newaxis, :]) * np.sin(Theta_diff)
    M = -H
    L = N

    # Adjust diagonal elements
    H -= np.diag(H.sum(axis=1) - np.diag(H))
    N += np.diag(N.sum(axis=1) - np.diag(N))
    M += np.diag(M.sum(axis=1) - np.diag(M))
    L -= np.diag(L.sum(axis=1) - np.diag(L))

    # Remove slack bus rows/columns
    H = H[np.ix_(non_slack, non_slack)]
    N = N[np.ix_(non_slack, non_slack)]
    M = M[np.ix_(non_slack, non_slack)]
    L = L[np.ix_(non_slack, non_slack)]

    # Construct full Jacobian
    J = np.block([[H, N], [M, L]])

    return J
