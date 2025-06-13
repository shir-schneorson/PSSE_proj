import numpy as np


def GN_se(z, v, slk_bus, h_ac, nb, tol= 1e-10, max_iter=500, x0=None):

    if x0 is None:
        T = np.zeros(nb)
        T[slk_bus[0]] = slk_bus[1]
        V = np.ones(nb)
    else:
        T = x0[:nb]
        T[slk_bus[0]] = slk_bus[1]
        V = x0[nb:]

    x = np.r_[T, V]
    R = np.diag(1. / v)
    eps = np.inf
    converged = False

    for iter in range(max_iter):
        z_est, J = h_ac.estimate(V, T)
        delta_z = z - z_est
        J = np.delete(J, slk_bus[0], axis=1)

        JT_R_J = J.T @ R @ J
        cond_number = np.linalg.cond(JT_R_J)
        if cond_number > 1e8:
            break

        delta_x = np.linalg.lstsq(J.T @ R @ J, J.T @ R @ delta_z, rcond=None)[0]
        delta_x = np.insert(delta_x, slk_bus[0], 0)

        eps = np.linalg.norm(delta_x, np.inf)

        x = x + delta_x
        T = x[:nb]
        V = x[nb:]

        if eps <= tol:
            converged = True
            break

    return T, V, eps, iter, converged
