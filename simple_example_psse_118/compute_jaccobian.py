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
    V_T, theta = V[None, :], theta[:, None]

    theta_diff = theta - theta.T
    V_outer = V_T.T @ V_T

    G_sin_theta, G_cos_theta = G * np.sin(theta_diff), G * np.cos(theta_diff)
    B_sin_theta, B_cos_theta = B * np.sin(theta_diff), B * np.cos(theta_diff)

    dP_inj_dTheta = V_outer * (G_sin_theta - B_cos_theta)
    dP_inj_dTheta -= np.diag(np.sum(dP_inj_dTheta, axis=1))
    dP_inj_dV = V * (G_cos_theta + B_sin_theta)
    dP_inj_dV += np.diag(np.sum(V_T * (G_cos_theta + B_sin_theta), axis=1))

    dQ_inj_dTheta = V_outer * (-G_cos_theta - B_sin_theta)
    dQ_inj_dTheta -= np.diag(np.sum(dQ_inj_dTheta, axis=1))
    dQ_inj_dV = V * (G_sin_theta - B_cos_theta)
    dQ_inj_dV += np.diag(np.sum(V_T * (G_cos_theta + B_sin_theta), axis=1))

    J11 = dP_inj_dTheta[:, non_slack]
    J12 = dP_inj_dV
    J21 = dQ_inj_dTheta[:, non_slack]
    J22 = dQ_inj_dV

    J = np.block([[J11, J12], [J21, J22]])

    return J
