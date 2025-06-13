import torch
from tqdm import tqdm


def GN_se_torch(z, v, slk_bus, h_ac, nb, tol=1e-10, max_iter=500, x0=None):
    device = z.device  # Ensure tensors stay on the same device

    if x0 is None:
        T = torch.zeros(nb, device=device)
        T[slk_bus[0]] = slk_bus[1]
        V = torch.ones(nb, device=device)
    else:
        T = x0[:nb].clone()
        T[slk_bus[0]] = slk_bus[1]
        V = x0[nb:].clone()

    x = torch.cat([T, V])
    R = torch.diag(1.0 / v)
    eps = float('inf')
    converged = False

    for iteration in tqdm(range(max_iter)):
        T.requires_grad_(True)
        V.requires_grad_(True)

        z_est, J_anlystic = h_ac.estimate(V, T)  # z_est shape: (m,)
        delta_z = z - z_est

        # Jacobian computation
        J_rows = []
        for i in range(z_est.shape[0]):
            grad = torch.autograd.grad(z_est[i], (T, V), retain_graph=True, allow_unused=True)
            grad_T = grad[0] if grad[0] is not None else torch.zeros_like(T)
            grad_V = grad[1] if grad[1] is not None else torch.zeros_like(V)
            J_rows.append(torch.cat([grad_T, grad_V]))

        J = torch.stack(J_rows)  # shape (m, 2nb)
        J = J_anlystic
        # assert torch.allclose(J, J_anlystic), f'iteration {iteration} failed'
        J = torch.cat([J[:, :slk_bus[0]], J[:, slk_bus[0]+1:]], dim=1)  # Remove slack angle column

        JT_R = J.T @ R
        lhs = JT_R @ J
        rhs = JT_R @ delta_z

        delta_x_reduced = torch.linalg.solve(lhs, rhs)
        delta_x = torch.cat([
            delta_x_reduced[:slk_bus[0]],
            torch.tensor([0.0], device=device),
            delta_x_reduced[slk_bus[0]:]
        ])

        eps = torch.norm(delta_x, p=float('inf')).item()

        x = x + delta_x
        T = x[:nb]
        V = x[nb:]

        if eps <= tol:
            converged = True
            break

    return T, V, eps, iteration, converged
