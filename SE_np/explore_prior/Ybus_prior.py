import numpy as np

from SE_np.net_preprocess.process_net_data import parse_ieee_mat, System


def make_topology_prior(sys, alpha_T=1.0, alpha_V=1.0):
    nb = sys.nb
    slk = int(sys.slk_bus[0])

    ImYbus = np.imag(sys.Ybus)
    B = 0.5 * (ImYbus + ImYbus.T)
    L = -B

    ang_mask = np.ones(nb, dtype=bool)
    ang_mask[slk] = False
    Lang = L[np.ix_(ang_mask, ang_mask)]

    # pq = np.where(sys.bus.bus_type.values == 1)[0]
    pq = ang_mask
    Lvol = L[np.ix_(pq, pq)]

    Lang = alpha_T * Lang
    Lvol = alpha_V * Lvol

    Prec = np.block([
        [Lang, np.zeros((Lang.shape[0], Lvol.shape[0]))],
        [np.zeros((Lvol.shape[0], Lang.shape[0])), Lvol]
    ])

    Cov = np.linalg.pinv(Prec)
    ang_idx_full = np.arange(0, nb, dtype=int)
    volt_idx_full = nb + np.arange(0, nb, dtype=int)

    T_red_idx_full = ang_idx_full[ang_mask]
    V_red_idx_full = volt_idx_full[pq]

    red_full_idx = np.concatenate([T_red_idx_full, V_red_idx_full])

    return Cov, Prec, red_full_idx

def get_sigma_points_and_weights(mu, cov, idx):
    n = len(idx)
    ro = (1e-3**2 * n) - n
    # ro = n-3
    L = np.linalg.cholesky((n + ro) * cov)
    mu_red = mu[idx, np.newaxis]
    X_0 = mu_red
    X_i = mu_red + L
    X_n_i = mu_red - L
    sigma_points = np.c_[X_0, X_i, X_n_i].T
    W = np.ones(2 * n + 1)
    W[0] =  ro / (n + ro)
    W[1:] = 1 / (2*(n + ro))

    return sigma_points, W

def full_state(x, mu, idx):
    x_full = mu.copy()
    x_full[idx] = x

    # return (x_full + np.pi) % (2*np.pi) - np.pi
    return x_full

def lamda(x, sys, pq, ii):
    T, V = x[:sys.nb], x[sys.nb:]
    Vc = V * np.exp(1j * T)
    S = Vc * np.conj(sys.Ybus @ Vc)
    return np.concatenate((np.real(S[ii]), np.imag(S[pq])))

def propagate(sigma_points, sys, mu, idx):
    pq = np.where(sys.bus.bus_type.values == 1)[0]
    ii = np.where(sys.bus.bus_type.values != 3)[0]
    Yi = np.stack([lamda(full_state(xi, mu, idx), sys, pq, ii) for xi in sigma_points])
    return Yi

def compute_mean_transformed(transformed_sigma_points, W):
    return np.sum(W[:, np.newaxis] * transformed_sigma_points, axis=0)

def compute_cov_transformed(Y, y_mu, W):
    return np.sum([W[i] * np.outer(Y[i] - y_mu, Y[i] - y_mu) for i in range(len(Y))], axis=0)

def main():
    file = '../../nets/ieee118_186.mat'

    data = parse_ieee_mat(file)
    system_data = data['data']['system']
    sys = System(system_data)

    mu = np.r_[np.zeros(sys.nb), np.ones(sys.nb)]
    cov, prec, idx = make_topology_prior(sys, alpha_T=0.045, alpha_V=10.0)
    # samples = np.random.multivariate_normal(mu[idx], cov, 100000)
    # pls = propagate(samples, sys, mu, idx)
    # mu_y = np.mean(pls, axis=0)
    # cov_y = np.cov(pls, rowvar=False)
    sigma_points, W = get_sigma_points_and_weights(mu, cov, idx)
    Yi = propagate(sigma_points, sys, mu, idx)
    mu_y = compute_mean_transformed(Yi, W)
    cov_y = compute_cov_transformed(Yi, mu_y, W)
    return mu_y, cov_y

if __name__ == '__main__':

    mu_y, cov_y = main()
    np.save('mu_pl.npy', mu_y)
    np.save('cov_pl.npy', cov_y)