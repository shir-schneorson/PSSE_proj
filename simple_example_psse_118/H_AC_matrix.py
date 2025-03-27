import numpy as np


def H_AC_matrix(x, Ybus):
    """
    Computes AC power injections (P, Q) using matrix form.

    Parameters:
    x : ndarray
        State vector [theta; V] where theta (angles) and V (amplitudes) are (n,).
    Ybus : ndarray
        Bus admittance matrix (n x n).

    Returns:
    P : ndarray
        Active power injections (n,).
    Q : ndarray
        Reactive power injections (n,).
    """
    # Number of buses
    n = len(Ybus)

    # Extract voltage angles and magnitudes
    theta = x[:n]  # Voltage angles in radians
    V = x[n:]  # Voltage magnitudes

    # Compute complex voltage vector
    V_complex = V * np.exp(1j * theta)

    # Compute complex power injections: S = V * conj(Ybus @ V)
    S = V_complex * np.conj(Ybus @ V_complex)

    # Extract active (P) and reactive (Q) power injections
    P = S.real
    Q = S.imag

    return P, Q
