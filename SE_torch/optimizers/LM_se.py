from math import sqrt

import torch
from tqdm import tqdm

from SE_torch.optimizers.base_optimizer import SEOptimizer

DEFAULT_NB = 118
SCALE_V = 0.025
SCALE_T = sqrt(torch.pi) / 2

def df_dV(V, H, z):
    z_est = torch.einsum('lij,ji->l', H, V)
    residuals = z_est - z
    return torch.tensordot(residuals, H, dims=(0, 0))


def load_prior(**kwargs):
    use_prior = kwargs.get('use_prior', False)
    reg_scale = kwargs.get('reg_scale', iter(torch.linspace(1, 1e-1, steps=50)))
    nb = kwargs.get('nb', DEFAULT_NB)
    m = kwargs.get('m',torch.concat([torch.zeros(nb),
                                     torch.ones(nb)])).to(torch.float64)
    Q = kwargs.get('Q', torch.diag(
        torch.concat([torch.ones(nb) * SCALE_T,
                      torch.ones(nb) * SCALE_V]))).to(torch.float64)
    slk_idx = kwargs.get('slk_idx', 0)

    if len(m) == nb * 2 - 1:
        m_red = m
        Q_red = Q
    else:
        mask = torch.ones_like(m, dtype=torch.bool, device=m.device)
        mask[slk_idx] = False

        m_red = m[mask]
        Q_red = Q[mask][:, mask]

    Q_inv = torch.linalg.pinv(Q_red)
    L = torch.linalg.cholesky(Q_red)
    L = torch.linalg.pinv(L)

    return use_prior, m_red, Q_inv, L, reg_scale


class LM_se(SEOptimizer):
    def __init__(self, **kwargs):
        super(LM_se, self).__init__(**kwargs)
        self.xtol = kwargs.get('xtol', 1e-10)
        self.ftol = kwargs.get('ftol', 1e-8)
        self.max_iter = int(kwargs.get('max_iter', 1e+6))
        self.verbose = kwargs.get('verbose', True)
        self.use_prior, self.m_red, self.Q_inv, self.L, self.reg_scale = load_prior(**kwargs)
        self.lamda = 1
        self.sigma = kwargs.get('sigma', 0.1)
        self.delta_init = kwargs.get('delta_init', 0.004)

        self.z = None
        self.v = None
        self.R = None
        self.h_ac = None
        self.slk_bus = None
        self.nb = None
        self.norm_H = None


    def set_reg_scale(self, new_reg_scale):
        self.reg_scale = float(new_reg_scale)
        if self.Q_inv is not None:
            self.L = self.reg_scale * self.Q_inv

    def _remove_slack_col(self, J):
        s = int(self.slk_bus[0])
        return torch.cat([J[:, :s], J[:, s+1:]], dim=1)

    def _insert_zero_at_slack(self, vec):
        s = int(self.slk_bus[0])
        return torch.cat([vec[:s], torch.zeros(1, device=vec.device, dtype=vec.dtype), vec[s:]], dim=0)

    def compute_J(self, x_full):
        T = x_full[:self.nb]
        V = x_full[self.nb:]
        J = self.h_ac.jacobian(T, V)
        J = -(self.R @ J) / self.norm_H[:, None]
        J = self._remove_slack_col(J)
        if self.use_prior:
            J = torch.cat([J, self.lamda * self.L], dim=0)
        return J

    def compute_f(self, x_full):
        T = x_full[:self.nb]
        V = x_full[self.nb:]
        z_est = self.h_ac.estimate(T, V)
        f_meas = (self.R @ (self.z - z_est)) / self.norm_H

        if self.use_prior:
            s = int(self.slk_bus[0])
            x_red = torch.cat([x_full[:s], x_full[s+1:]], dim=0)
            f_prior = self.L @ (x_red - self.m_red)
            return torch.cat([f_meas, self.lamda * f_prior], dim=0)

        return f_meas

    def compute_D(self, J, D_prev=None):
        JTJ = J.T @ J
        d = torch.sqrt(torch.diag(JTJ))
        if D_prev is not None:
            d_prev = torch.diag(D_prev)
            d = torch.maximum(d_prev, d)
        return torch.diag(d)

    def compute_delta(self, x, rho, D, step_size, lam, delta_prev=None):
        norm_Dp = torch.norm(D @ step_size)
        if delta_prev is None:
            delta_prev = norm_Dp

        if ((0.25 < rho) and (rho < 0.75) and (lam == 0)) or (rho >= 0.75):
            return 2.0 * norm_Dp

        norm_f = torch.norm(self.compute_f(x))
        norm_fp = torch.norm(self.compute_f(x + self._insert_zero_at_slack(step_size)))
        if norm_fp <= norm_f:
            mu = 0.5
        elif norm_fp > 10.0 * norm_f:
            mu = 0.1
        else:
            norm_Jp = torch.norm(self.compute_J(x) @ step_size)
            gamma = -((norm_Jp / norm_f) ** 2 + (torch.sqrt(torch.as_tensor(lam, dtype=norm_f.dtype, device=norm_f.device)) * norm_Dp / norm_f) ** 2)
            mu = 0.5 * gamma / (gamma + (0.5 * (1 - (norm_fp / norm_f) ** 2)))
        mu = max(0.1, min(0.5, float(mu)))
        return mu * delta_prev

    def compute_step_size(self, f, J, D, delta=None):
        p = self.p_lam(torch.as_tensor(0.0, dtype=J.dtype, device=J.device), f, J, D)
        if (delta is None) or (torch.norm(D @ p) <= (1.0 + self.sigma) * delta):
            return p, 0.0
        lam = self.compute_lam(f, J, D, delta)
        return self.p_lam(lam, f, J, D), lam

    def p_lam(self, lam, f, J, D):
        JTJ = J.T @ J
        g = -J.T @ f
        DTD = D.T @ D
        p = torch.linalg.lstsq(JTJ + lam * DTD, g).solution
        return p

    def compute_lam(self, f, J, D, delta, tol=1e-10, max_iter=50):
        D_inv = torch.linalg.inv(D)
        J_tilda = J @ D_inv
        U, S, _ = torch.linalg.svd(J_tilda, full_matrices=False)
        z = (U.T @ f)[:S.shape[0]]

        def phi_lam(lam):
            return torch.norm((S * z) / (S**2 + lam)) - delta

        def grad_phi_lam(lam):
            denom = (S**2 + lam)
            r = (S * z) / denom
            nr = torch.norm(r)
            if nr == 0:
                return 0.0
            numer = torch.sum((S**2 * (torch.abs(z) ** 2)) / (denom ** 3))
            return -numer / nr

        lam = 0.0
        phi = phi_lam(lam)
        if phi < tol:
            return lam

        grad = grad_phi_lam(lam)
        if grad == 0:
            grad = torch.as_tensor(-1e-16, dtype=J.dtype, device=J.device)

        l = -phi / grad
        JDinv = J @ D_inv
        u = torch.norm(JDinv.conj().T @ f) / max(delta, 1e-16)

        for _ in range(max_iter):
            lam = lam if l < lam < u else max(1e-3 * u, (max(l, 0.0) * u) ** 0.5)
            phi = phi_lam(lam)
            grad = grad_phi_lam(lam) or -1e-16
            u = lam if phi < 0 else u
            l = max(l, lam - (phi / grad))
            lam = lam - (((phi + delta) / max(delta, 1e-16)) * (phi / grad))
            if abs(phi) < tol:
                break

        return lam

    def compute_rho(self, x, D, step_size, alpha):
        norm_f = torch.norm(self.compute_f(x))
        norm_fp = torch.norm(self.compute_f(x + self._insert_zero_at_slack(step_size)))
        norm_Jp = torch.norm(self.compute_J(x) @ step_size)
        norm_Dp = torch.norm(D @ step_size)
        num = 1.0 - (norm_fp / norm_f) ** 2
        if num < 0:
            return 0.0
        denom = (norm_Jp / norm_f) ** 2 + 2.0 * ((torch.sqrt(torch.as_tensor(alpha, dtype=norm_f.dtype, device=norm_f.device)) * norm_Dp) / norm_f) ** 2
        return float(num / denom)

    def __call__(self, x0, z, v, slk_bus, h_ac, nb, norm_H=None):
        x = x0.clone()
        self.z = z
        self.v = v
        self.R = torch.diag(1./ torch.sqrt(v))
        self.h_ac = h_ac
        self.slk_bus = slk_bus
        self.nb = nb
        self.norm_H = norm_H if norm_H is not None else torch.ones_like(z)

        J = self.compute_J(x)
        f = self.compute_f(x)
        D = self.compute_D(J)
        delta = self.delta_init
        converged = False
        it = 0
        for it in tqdm(range(self.max_iter), desc=f"Optimizing with LM", leave=False, colour='green'):
            step_size, lam = self.compute_step_size(f, J, D, delta)
            rho = self.compute_rho(x, D, step_size, lam)

            if (rho > 1e-4) or (torch.norm(step_size, p=float('inf')) <= self.xtol):
                x = x + self._insert_zero_at_slack(step_size)
                J = self.compute_J(x)
                f = self.compute_f(x)

                if torch.norm(step_size, p=float('inf')) <= self.xtol:
                    converged = True
                    break

                D = self.compute_D(J, D_prev=D)

            delta = self.compute_delta(x, rho, D, step_size, lam, delta_prev=delta)
            eps = torch.norm(step_size, p=float('inf')).item()
            loss = torch.linalg.norm(f) ** 2
            self.lamda = next(self.reg_scale, 1e-1)

            if self.verbose:
                print(f'LM - iter: {it}, delta: {eps:.5e}, loss: {loss.item():.4f}')

        T, V = x[:nb], x[nb:]
        return x, T, V, converged, it