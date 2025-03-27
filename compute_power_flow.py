import numpy as np

def compute_power_flow(theta, V, G, B):
    """
    Computes active (P) and reactive (Q) power injections.

    Parameters:
    theta : ndarray
        Voltage angles (n,).
    V : ndarray
        Voltage magnitudes (n,).
    G : ndarray
        Conductance matrix (n x n).
    B : ndarray
        Susceptance matrix (n x n).

    Returns:
    P : ndarray
        Active power injections (n,).
    Q : ndarray
        Reactive power injections (n,).
    """
    n = len(V)
    P = np.zeros(n)
    Q = np.zeros(n)

    # Compute all pairwise angle differences
    Theta_diff = theta[:, np.newaxis] - theta[np.newaxis, :]

    # Compute power injections using vectorized matrix form
    P = V * np.sum(V[np.newaxis, :] * (G * np.cos(Theta_diff) + B * np.sin(Theta_diff)), axis=1)
    Q = V * np.sum(V[np.newaxis, :] * (G * np.sin(Theta_diff) - B * np.cos(Theta_diff)), axis=1)

    return P, Q