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

    theta_diff = theta[:, np.newaxis] - theta[np.newaxis, :]

    V_outer = V[:, np.newaxis] * V[np.newaxis, :]
    J11 = V_outer * (G * np.sin(theta_diff) - B * np.cos(theta_diff))
    J12 = V[:, np.newaxis] * (G * np.cos(theta_diff) + B * np.sin(theta_diff))
    J21 = V_outer * (G * -np.cos(theta_diff) + B * -np.sin(theta_diff))
    J22 = V[:, np.newaxis] * (G * np.sin(theta_diff) - B * np.cos(theta_diff))

    np.fill_diagonal(J11, np.sum(-J11, axis=1) + J11.diagonal())
    np.fill_diagonal(J12, np.sum(V[np.newaxis, :] * (G * np.cos(theta_diff) + B * np.sin(theta_diff)), axis=1))
    np.fill_diagonal(J21, np.sum(-J21, axis=1) + J21.diagonal())
    np.fill_diagonal(J22, np.sum(V[np.newaxis, :] * (G * np.sin(theta_diff) - B * np.cos(theta_diff)), axis=1))

    J12 += np.diag(3 * V * G.diagonal())
    J22 += np.diag(-3 * V * B.diagonal())


    J11 = J11[np.ix_(non_slack, non_slack)]
    J12 = J12[np.ix_(non_slack, non_slack)]
    J21 = J21[np.ix_(non_slack, non_slack)]
    J22 = J22[np.ix_(non_slack, non_slack)]

    J = np.block([[J11, J12], [J21, J22]])

    return J
