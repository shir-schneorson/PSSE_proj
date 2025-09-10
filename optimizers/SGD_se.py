from tqdm import tqdm

import numpy as np


def step_size(u0, H, z, norm_H=None, V_lower=.95, V_upper=1.05, c=4, n_samples=20):
    rng = np.random.default_rng()

    u = u0.copy()
    if u.ndim == 1:
        u = u[:, np.newaxis]
    V0 = u @ np.conj(u).T

    n = V0.shape[0]
    grad_V0 = grad_f(V0, H, z, norm_H)
    V_lower = V_lower * np.ones(n)
    V_upper = V_upper * np.ones(n)

    def draw_V():
        theta = np.pi * .35
        T = np.random.normal(0, theta / 3, n)

        V = np.random.normal(1, 0.05 / 3, n)
        v = V * np.exp(1j * T)
        # v = rng.standard_normal(n) + 1j * rng.standard_normal(n) if np.iscomplexobj(V0) \
        #     else rng.standard_normal(n)
        # v /= np.linalg.norm(v) + 1e-12
        #
        # mags_sq = rng.uniform(V_lower ** 2, V_upper ** 2, size=n)
        # v = np.sqrt(mags_sq) * v / np.abs(v)
        return np.outer(v, v.conj())

    ratios = []
    for _ in range(n_samples):
        V = draw_V()
        num = np.linalg.norm(grad_V0 - grad_f(V, H, z, norm_H), "fro")
        den = np.linalg.norm(V0 - V, "fro") + 1e-16
        ratios.append(num / den)

    M_hat = max(ratios)

    eta =  1 / ((M_hat * np.linalg.norm(V0, 2) + np.linalg.norm(grad_V0, 2)) * c)
    return eta


def SGD_se_obj(u, H, z, norm_H=None):
    Hu = np.tensordot(H, u, axes=([2], [0]))
    residuals = np.einsum('j,lj->l', np.conj(u), Hu) - z
    if norm_H is not None:
        residuals = residuals / norm_H
    return 0.5 * np.sum(residuals ** 2)


def grad_g(u, H, z, norm_H=None):
    Hu = np.tensordot(H, u, axes=([2], [0]))
    residuals = np.einsum('j,lj->l', np.conj(u), Hu) - z
    if norm_H is not None:
        Hu = Hu / norm_H[:, np.newaxis]
        residuals = residuals / norm_H
    grad = 2 *  Hu.T @ residuals
    return grad.astype(u.dtype)


def grad_f( V, H, z, norm_H=None):
    residuals = np.einsum('lij,ji->l', H, V) - z
    if norm_H is not None:
        residuals = residuals / norm_H
        H = H / norm_H[:, np.newaxis, np.newaxis]
    grad = np.tensordot(residuals, H, axes=(0, 0))
    return grad.astype(V.dtype)


def FGD_se(u0, measurements, H, eta, norm_H, slk_bus, AGD_update=False, K=500, tol=2e-4, tol_obj=5e-4):
    u = u0.copy()

    converged = False
    grad_u = grad_g(u, H, measurements, norm_H)
    grad_u[slk_bus[0]] = 0
    u_prev = u
    u_curr = u - (eta * grad_u)
    u_list = [u_prev, u_curr]

    obj_curr = SGD_se_obj(u_curr, H, measurements, norm_H)

    optimizer = 'AGD' if AGD_update else 'FGD'

    for k in tqdm(range(2, K + 1), desc=f'Optimizing with {optimizer}', leave=False, colour='green'):
        if AGD_update:
            momentum = (k - 1) / (k + 2)
            u_p = u_curr + momentum * (u_curr - u_prev)
        else:
            u_p = u_curr
        grad_u_p = grad_g(u_p, H, measurements, norm_H)
        grad_u_p[slk_bus[0]] = 0
        delta_u = eta * grad_u_p
        u_next = u_p - delta_u

        obj_next = SGD_se_obj(u_next, H, measurements, norm_H)
        u_list.append(u_next.copy())
        eps_iter = np.linalg.norm(delta_u, np.inf)
        eps_obj = abs(obj_next - obj_curr)

        u_prev, u_curr = u_curr, u_next
        obj_curr = obj_next
        if (eps_iter <= tol) and (eps_obj <= tol_obj):
            converged = True
            break


    return u_curr, u_list, converged, k


