import numpy as np
from tqdm import tqdm


def GN_se(x0, z, v, slk_bus, h_ac, nb, norm_H=None, tol= 1e-8, max_iter=500, prefix=''):

    x = x0.copy()
    T = x[:nb].copy()
    T[slk_bus[0]] = slk_bus[1]
    V = x[nb:].copy()

    is_norm = '-norm' if norm_H is not None else ''
    if norm_H is None:
        norm_H = np.ones_like(z)

    R = np.diag(1. / v)

    converged = False
    x_list = [x]

    for k in tqdm(range(max_iter), desc=f'Optimizing{prefix} with GN{is_norm}', leave=False, colour='green'):
        z_est, J = h_ac.estimate(V, T)
        delta_z = (z - z_est)/ norm_H
        J = np.delete(J, slk_bus[0], axis=1) / norm_H[:, None]

        JT_R_J = J.T @ R @ J
        try:
            delta_x = np.linalg.lstsq(JT_R_J, J.T @ R @ delta_z, rcond=None)[0]
        except np.linalg.LinAlgError:
            break

        delta_x = np.insert(delta_x, slk_bus[0], 0)

        eps = np.linalg.norm(delta_x, np.inf)
        err = np.linalg.norm(delta_z)

        x = x + delta_x
        x_list.append(x.copy())
        T = x[:nb]
        V = x[nb:]

        if eps <= tol:
            converged = True
            break

    return x, x_list, converged, k
