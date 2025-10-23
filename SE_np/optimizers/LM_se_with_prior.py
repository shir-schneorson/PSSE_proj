import numpy as np
from tqdm import tqdm


def df_dV(V, H, z):
    z_est = np.einsum('lij,ji->l', H, V)
    residuals = z_est - z
    return np.tensordot(residuals, H, axes=(0, 0))


class LMOptimizerSE:
    def __init__(self, h_ac, z, v, slk_bus, nb, delta_init,
                 norm_H=None, sigma=0.1, K=500, ftol=1e-8, xtol=1e-8,
                 m=None, Q=None, reg_scale=1.0, prefix='-norm'
    ):
        """
        Adds a Gaussian prior x ~ N(m, Q) on the *full* state x = [T(0:nb), V(nb:)].
        Internally we remove the slack element from the angle block so the prior
        aligns with the reduced variable vector used by LM (the one with the slack
        column removed in J).
        """
        self.h_ac = h_ac
        self.z = z
        self.v = v
        self.norm_H = norm_H if norm_H is not None else np.ones_like(z)
        self.R = np.diag(1 / np.sqrt(v))
        self.slk_bus = slk_bus
        self.nb = nb
        self.K = K
        self.xtol = xtol
        self.ftol = ftol
        self.sigma = sigma
        self.delta_init = delta_init
        self.prefix = prefix if prefix != '-norm' or norm_H is not None else ''

        # ---- Prior on full state; reduce by removing slack angle ----
        self.reg_scale = float(reg_scale)
        self._prior_enabled = (m is not None) and (Q is not None)
        self.m_red = None
        self.L = None

        if self._prior_enabled:
            m = np.asarray(m, dtype=float)
            Q = np.asarray(Q, dtype=float)

            n_full = m.shape[0]
            slk_idx = int(self.slk_bus[0])

            mask = np.ones(n_full, dtype=bool)
            mask[slk_idx] = False

            m_red = m[mask]
            Q_red = Q[np.ix_(mask, mask)]

            Q_inv = np.linalg.inv(Q_red + 1e-10 * np.eye(len(Q_red)))

            self.m_red = m_red
            self.Q_inv = Q_inv
            self.L = self.reg_scale * Q_inv

    def set_reg_scale(self, new_reg_scale):
        self.reg_scale = new_reg_scale
        self.L = self.reg_scale * self.Q_inv

    def compute_J(self, x_full):
        T = x_full[:self.nb]
        V = x_full[self.nb:]
        _, J = self.h_ac.estimate(V, T)
        J = (- self.R @ J) / self.norm_H[:, np.newaxis]
        J = np.delete(J, self.slk_bus[0], axis=1)

        if self._prior_enabled:
            J = np.vstack([J, self.L])
        return J

    def compute_f(self, x_full):
        T = x_full[:self.nb]
        V = x_full[self.nb:]
        z_est, _ = self.h_ac.estimate(V, T)
        f_meas = (self.R @ (self.z - z_est)) / self.norm_H

        if self._prior_enabled:
            x_red = np.delete(x_full, self.slk_bus[0])
            f_prior = self.L @ (x_red - self.m_red)
            return np.concatenate([f_meas, f_prior], axis=0)

        return f_meas

    def compute_D(self, J, D_prev=None):
        JTJ = J.T @ J
        d = np.sqrt(np.diag(JTJ))
        if D_prev is not None:
            d_prev = np.diag(D_prev)
            d = np.max([d_prev, d], axis=0)
        return np.diag(d)

    def compute_delta(self, x, rho, D, step_size, lam, delta_prev=None):
        norm_Dp = np.linalg.norm(D @ step_size)
        if delta_prev is None:
            delta_prev = norm_Dp

        if ((.25 < rho < .75) and (lam == 0)) or (rho >= .75):
            return 2 * norm_Dp

        norm_f = np.linalg.norm(self.compute_f(x))
        norm_fp = np.linalg.norm(self.compute_f(x + np.insert(step_size, self.slk_bus[0], 0)))
        if norm_fp <= norm_f:
            mu = .5
        elif norm_fp > 10 * norm_f:
            mu = .1
        else:
            norm_Jp = np.linalg.norm(self.compute_J(x) @ step_size)
            gamma = - ((norm_Jp / norm_f) ** 2 + (np.sqrt(lam) * norm_Dp / norm_f) ** 2)
            mu = .5 * gamma / (gamma + (.5 * (1 - (norm_fp / norm_f) ** 2)))
        mu = max(.1, min(.5, mu))
        return mu * delta_prev

    def compute_step_size(self, f, J, D, delta=None):
        p = self.p_lam(0, f, J, D)
        if delta is None or np.linalg.norm(D @ p) <= (1 + self.sigma) * delta:
            return p, 0.0
        lam = self.compute_lam(f, J, D, delta)
        return self.p_lam(lam, f, J, D), lam

    def p_lam(self, lam, f, J, D):
        JTJ = J.T @ J
        g = -J.T @ f
        DTD = D.T @ D
        return np.linalg.lstsq(JTJ + lam * DTD, g, rcond=None)[0]

    def compute_lam(self, f, J, D, delta, tol=1e-10, max_iter=50):
        D_inv = np.linalg.inv(D)
        J_tilda = J @ D_inv
        U, S, _ = np.linalg.svd(J_tilda, full_matrices=False)
        z = (U.T @ f)[:len(S)]

        def phi_lam(lam):
            return np.linalg.norm((S * z) / (S**2 + lam)) - delta

        def grad_phi_lam(lam):
            denom = (S**2 + lam)
            r = (S * z) / denom
            nr = np.linalg.norm(r)
            if nr == 0:
                return 0.0
            numer = np.sum((S**2 * (np.abs(z)**2)) / (denom**3))
            return -numer / nr

        lam = 0.0
        phi = phi_lam(lam)
        if phi < tol:
            return lam
        grad = grad_phi_lam(lam) or -1e-16
        l = -phi / grad
        u = np.linalg.norm((J @ D_inv).conj().T @ f) / max(delta, 1e-16)

        for _ in range(max_iter):
            lam = lam if l < lam < u else max(1e-3 * u, np.sqrt(max(l, 0) * u))
            phi = phi_lam(lam)
            grad = grad_phi_lam(lam) or -1e-16
            u = lam if phi < 0 else u
            l = max(l, lam - (phi / grad))
            lam = lam - (((phi + delta) / max(delta, 1e-16)) * (phi / grad))
            if abs(phi) < tol:
                break
        return lam

    def compute_rho(self, x, D, step_size, alpha):
        norm_f = np.linalg.norm(self.compute_f(x))
        norm_fp = np.linalg.norm(self.compute_f(x + np.insert(step_size, self.slk_bus[0], 0)))
        norm_Jp = np.linalg.norm(self.compute_J(x) @ step_size)
        norm_Dp = np.linalg.norm(D @ step_size)
        num = 1 - (norm_fp / norm_f) ** 2
        if num < 0:
            return 0.0
        denom = ((norm_Jp / norm_f) ** 2) + (2 * ((np.sqrt(alpha) * norm_Dp) / norm_f) ** 2)
        return num / denom

    def optimize(self, x0):
        x = x0.copy()
        J = self.compute_J(x)
        f = self.compute_f(x)
        D = self.compute_D(J)
        delta = self.delta_init
        converged = False
        x_list = [x]

        for k in tqdm(range(self.K), desc=f"Optimizing with LM{self.prefix}", leave=False, colour='green'):
            step_size, lam = self.compute_step_size(f, J, D, delta)
            rho = self.compute_rho(x, D, step_size, lam)

            if rho > 1e-4 or np.linalg.norm(step_size, np.inf) <= self.xtol:
                x = x + np.insert(step_size, self.slk_bus[0], 0)
                J = self.compute_J(x)
                f = self.compute_f(x)
                x_list.append(x.copy())

                if np.linalg.norm(step_size, np.inf) <= self.xtol:
                    converged = True
                    break

                D = self.compute_D(J, D_prev=D)

            delta = self.compute_delta(x, rho, D, step_size, lam, delta_prev=delta)

        return x, x_list, converged, k
