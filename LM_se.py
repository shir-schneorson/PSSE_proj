import numpy as np
from tqdm import tqdm


def df_dV(V, H ,z):
    z_est = np.einsum('lij,ji->l', H, V)
    residuals = z_est - z
    return np.tensordot(residuals, H, axes=(0, 0))

class LMOptimizerSE:
    def __init__(self, h_ac, z, v, slk_bus, nb, delta_init, norm_H=None, sigma=0.1, K=500, ftol=1e-8, xtol=1e-8):
        self.h_ac = h_ac
        self.z = z
        self.v = v
        self.R = np.diag(1 / np.sqrt(v))
        self.slk_bus = slk_bus
        self.nb = nb
        self.norm_H = norm_H if norm_H is not None else np.ones_like(z)
        self.K = K
        self.xtol = xtol
        self.ftol = ftol
        self.sigma = sigma
        self.delta_init = delta_init

    def compute_J(self, x):
        T = x[:self.nb]
        V = x[self.nb:]
        _, J = self.h_ac.estimate(V, T)
        return (- self.R @ J) / self.norm_H[:, np.newaxis]

    def compute_f(self, x):
        T = x[:self.nb]
        V = x[self.nb:]
        z_est, _ = self.h_ac.estimate(V, T)
        return (self.R @ (self.z - z_est)) / self.norm_H

    def compute_D(self, J, D_prev=None):
        JT_R_J = J.T @ J
        d = np.sqrt(np.diag(JT_R_J))
        if D_prev is not None:
            d_prev = np.diag(D_prev)
            d = np.max([d_prev, d], axis=0)
        return np.diag(d)

    def compute_delta(self,x, rho,  D, step_size, lam, delta_prev=None):
        norm_Dp = np.linalg.norm(D @ step_size)
        if delta_prev is None:
            delta_prev = norm_Dp

        if ((.25 < rho < .75) and (lam == 0)) or (rho >= .75):
            return 2 * norm_Dp

        norm_f = np.linalg.norm(self.compute_f(x))
        norm_fp = np.linalg.norm(self.compute_f(x + step_size))
        if norm_fp <= norm_f:
            mu = .5
        elif norm_fp > 10 * norm_f:
            mu = .1
        else:
            norm_Jp = np.linalg.norm(self.compute_J(x) @ step_size)
            gamma = - ((norm_Jp / norm_f) ** 2 + (np.sqrt(lam) * norm_Dp / norm_f) ** 2)
            mu = .5 * gamma / (gamma + (.5 * (1 - (norm_fp / norm_f) ** 2)))

        mu = .1 if mu < .1 else mu
        mu = .5 if mu > .5 else mu

        return mu * delta_prev

    def compute_step_size(self, f, J, D, delta=None):
        p = self.p_lam(0, f, J, D)
        if delta is None or np.linalg.norm(D @ p) <= (1 + self.sigma) * delta:
            step_size = p
            lam = 0
        else:
            lam = self.compute_lam( f, J, D, delta)
            step_size = self.p_lam(lam, f, J, D)

        return step_size, lam

    def p_lam(self, lam, f, J, D):
        JT_R_J = J.T @ J
        JT_R_f = -J.T @ f
        DTD = D.T @ D
        return np.linalg.lstsq(JT_R_J + lam * DTD, JT_R_f, rcond=None)[0]

    def compute_lam(self, f, J, D, delta, tol=1e-10, max_iter=50):
        D_inv = np.linalg.inv(D)
        J_tilda = J @ D_inv
        U, S, V = np.linalg.svd(J_tilda)
        z = (U.T @ f)[:len(S)]

        def phi_lam(lam):
            return np.linalg.norm((S * z[:len(S)]) / (S**2 + lam)) - delta

        def grad_phi_lam(lam):
            denom = (S ** 2 + lam)
            r = (S * z) / denom
            norm_r = np.linalg.norm(r)
            if norm_r == 0:
                return 0.0
            numer = np.sum((S ** 2 * np.abs(z) ** 2) / denom ** 3)
            return -numer / norm_r

        lam = 0
        phi = phi_lam(lam)
        if phi < tol:
            return lam
        grad_phi = grad_phi_lam(lam)
        l = -phi / grad_phi
        u = np.linalg.norm((J @ D_inv).conj().T @ f) / delta

        for _ in range(max_iter):
            lam = lam if l < lam < u else max(1e-3 * u, np.sqrt(l * u))
            phi = phi_lam(lam)
            grad_phi = grad_phi_lam(lam)
            u = lam if phi < 0 else u
            l = max(l, lam - (phi / grad_phi))
            lam = lam - (((phi + delta) / delta) * (phi / grad_phi))
            if abs(phi) < tol:
                return lam

        return lam


    def compute_rho(self, x, D, step_size, alpha):
        norm_f = np.linalg.norm(self.compute_f(x))
        norm_fp = np.linalg.norm(self.compute_f(x + step_size))
        norm_Jp = np.linalg.norm(self.compute_J(x) @ step_size)
        norm_Dp = np.linalg.norm(D @ step_size)
        num = 1 - (norm_fp / norm_f) ** 2
        if num < 0:
            return 0
        denom = ((norm_Jp / norm_f) ** 2) + (2 * ((np.sqrt(alpha) * norm_Dp) / norm_f) ** 2)
        rho = num / denom
        return rho


    def optimize(self, x0):

        x = x0.copy()
        J = self.compute_J(x)
        f = self.compute_f(x)
        D = self.compute_D(J)
        delta = self.delta_init
        converged = False

        x_list = [x]

        for _ in tqdm(range(self.K), desc="Optimizing with Levenberg-Marquardt"):

            step_size, lam = self.compute_step_size(f, J, D, delta)
            rho = self.compute_rho(x, D, step_size, lam)

            for _ in range(10):
                if rho > 1e-4:
                    break
                delta = self.compute_delta(x, rho, D, step_size, lam, delta_prev=delta)

                step_size, lam = self.compute_step_size(f, J, D, delta)
                rho = self.compute_rho(x, D, step_size, lam)

            x = x + step_size
            J = self.compute_J(x)
            f = self.compute_f(x)

            x_list.append(x.copy())

            if np.linalg.norm(step_size, np.inf) <= self.xtol:
                converged = True
                break

            delta = self.compute_delta(x, rho,  D, step_size, lam, delta_prev=delta)
            D = self.compute_D(J, D_prev=D)

        return x, x_list, converged


