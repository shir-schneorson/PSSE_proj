import numpy as np
from tqdm import tqdm


def GN_se(x0, z, v, slk_bus, h_ac, nb, norm_H=None, tol= 1e-8, max_iter=500, prefix=''):

    x = x0.copy()
    T = x[:nb].copy()
    # T[slk_bus[0]] = slk_bus[1]
    V = x[nb:].copy()

    if norm_H is None:
        norm_H = np.ones_like(z)

    R = np.diag(norm_H / v)

    eps = np.inf
    converged = False
    x_list = [x]

    for iter in tqdm(range(max_iter), desc=f'Optimizing {prefix} with GN'):
        z_est, J = h_ac.estimate(V, T)
        delta_z = (z - z_est)/ norm_H
        J = np.delete(J, slk_bus[0], axis=1) / norm_H[:, None]

        JT_R_J = J.T @ R @ J
        cond_number = np.linalg.cond(JT_R_J)
        if cond_number > 1e10:
            break

        delta_x = np.linalg.lstsq(J.T @ R @ J, J.T @ R @ delta_z, rcond=None)[0]
        delta_x = np.insert(delta_x, slk_bus[0], 0)

        eps = np.linalg.norm(delta_x, np.inf)

        x = x + delta_x
        x_list.append(x.copy())
        T = x[:nb]
        V = x[nb:]

        if eps <= tol:
            converged = True
            break

    return x, x_list, converged
