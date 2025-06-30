import numpy as np
from tqdm import tqdm


def sample_u_with_voltage_constraints(shape, V_min, V_max, theta):
    n, _ = shape
    theta *= np.pi

    T = np.random.uniform(0, theta, size=(n, 1))
    T[0] = 0  # Fix phase at reference bus

    V = np.random.uniform(V_min, V_max, size=(n, 1))
    V[0] = 1  # Fix voltage magnitude at reference bus

    return V * np.exp(1j * T)


def step_size(u0, H, z, V_min=.95, V_max=1.05, theta=.35, c=4, max_iter=10):
    if u0.ndim == 1:
        u0 = u0[:, np.newaxis]
    V0 = u0 @ np.conj(u0).T
    grad_V0 = df_dV(V0, H, z)
    norm_grad_V0 = np.linalg.norm(grad_V0, 2)
    norm_V0 = np.linalg.norm(V0, 2)

    M_sum = 0
    count = 0

    for i in range(1, max_iter):
        u = sample_u_with_voltage_constraints(u0.shape, V_min, V_max, theta)
        V = u @ np.conj(u).T
        grad_V = df_dV(V, H, z)

        num = np.linalg.norm(grad_V0 - grad_V, 'fro')
        denom = np.linalg.norm(V0 - V, 'fro')

        if denom >= 1e-12:
            M_sum += num / denom
            count += 1

    M = M_sum / count if count > 0 else 1.0  # fallback to avoid instability
    eta = 1 / (c * (M * norm_V0 + norm_grad_V0))
    return eta

def SGD_se_obj(u, H, z):
    if u.ndim == 1:
        u = u[:, np.newaxis]
    UUH = u @ np.conj(u).T
    z_est = np.einsum('lij,ji->l', H, UUH)  # batched trace
    residual = z - z_est
    return 0.5 * np.dot(residual, residual)


def dg_du(u, H ,z):
    UUH = u @ np.conj(u).T
    df_UUH = df_dV(UUH, H, z)
    return 2 * df_UUH @ u


def df_dV(V, H ,z):
    z_est = np.einsum('lij,ji->l', H, V)
    residuals = z_est - z
    return np.tensordot(residuals, H, axes=(0, 0))


def FGD_se(u0, measurements, h_ac, eta, AGD_update=False, K=500, tol=2e-4, tol_obj=5e-4):
    u = u0.copy()
    if u.ndim == 1:
        u = u[:, np.newaxis]

    H = h_ac.H
    converged = False

    grad_u = dg_du(u, H, measurements)

    u_prev = u
    u_curr = u - eta * grad_u
    u_list = [u_prev, u_curr]

    obj_curr = SGD_se_obj(u_curr, H, measurements)

    optimizer = 'AGD' if AGD_update else 'FGD'

    for k in tqdm(range(2, K), desc=f'Optimizing with {optimizer}'):
        if AGD_update:
            momentum = (k - 1) / (k + 2)
            u_p = u_curr + momentum * (u_curr - u_prev)
        else:
            u_p = u_curr

        grad_u_p = dg_du(u_p, H, measurements)
        delta_u = eta * grad_u_p
        u_next = u_p - delta_u

        obj_next = SGD_se_obj(u_next, H, measurements)
        u_list.append(u_next.copy())
        eps_iter = np.linalg.norm(delta_u, np.inf)
        eps_obj = abs(obj_next - obj_curr)

        if eps_iter <= tol and eps_obj <= tol_obj:
            converged = True
            break

        u_prev, u_curr = u_curr, u_next
        obj_curr = obj_next

    return u_next, u_list, converged


