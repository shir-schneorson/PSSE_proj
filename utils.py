import numpy as np
from numpy._typing import NDArray

from init_net.power_flow_cartesian import H_AC as H_AC_cartesian
from init_net.power_flow_polar import H_AC as H_AC_polar
from optimizers.SGD_se import SGD_se_obj
from init_net.init_starting_point import init_start_point

file = "C:/Users/shirsc/PycharmProjects/PSSE_proj/nets/ieee30_41.mat"

def normalize_measurements(
    H: NDArray[np.complex128] | NDArray[np.float64],
    z: NDArray[np.complex128] | NDArray[np.float64]
) -> tuple[NDArray, NDArray, NDArray]:
    """
    Normalize each (z_l, H_l) pair by ||H_l||_F.

    Parameters
    ----------
    H : (L, n, n) ndarray
        Measurement matrices.
    z : (L,) ndarray
        Measurement scalars.

    Returns
    -------
    H_normalized : (L, n, n) ndarray
        Each H_l divided by ||H_l||_F.
    z_normalized : (L,) ndarray
        Each z_l divided by ||H_l||_F.
    """
    norms = np.linalg.norm(H, axis=(1, 2)) + 1e-12  # (L,), avoid divide-by-zero
    H_normalized = H / norms[:, None, None]
    z_normalized = z / norms
    return z_normalized, H_normalized, norms

# def normalize_measurements(measurements, H):
#     norm = np.linalg.norm(H, ord='fro', axis=(1, 2))
#
#     measurements_normalized = measurements / norm
#     H_normalized = H / norm[:, None, None]
#     return measurements_normalized, H_normalized, norm


def aggregate_meas_idx(meas_idx, meas_types):
    agg_meas_idx = {}
    last_idx = 0
    for k in meas_types:
        v = np.where(meas_idx.get(f'{k}_idx', []))[0]
        agg_meas_idx[k] = np.arange(last_idx, last_idx + len(v))
        last_idx += len(v)
    return agg_meas_idx


def generate_data(data, sys, branch, init_params, **kwargs):
    sys.slk_bus[1] = 0
    sys.slk_bus[2] = 1
    if kwargs.get('data_generator') is not None:
        data_generator = kwargs.get('data_generator')
        T_true, V_true = data_generator.sample(sys)
    else:
        T_true, V_true = init_start_point(sys, data, how='random', random_init=init_params)
    Vc_true = V_true * np.exp(1j * T_true)
    meas_idx = {}
    if kwargs.get('flow'):
        meas_idx['Pf_idx'] = np.r_[np.ones(len(branch.i)// 2), np.zeros(len(branch.i)// 2)].astype(bool)
        meas_idx['Qf_idx'] = np.r_[np.ones(len(branch.i)// 2), np.zeros(len(branch.i)// 2)].astype(bool)
    if kwargs.get('injection'):
        meas_idx['Pi_idx'] = np.ones(len(sys.bus)).astype(bool)
        meas_idx['Qi_idx'] = np.ones(len(sys.bus)).astype(bool)
    if kwargs.get('voltage'):
        meas_idx['Vm_idx'] = np.ones(len(sys.bus)).astype(bool)
    if kwargs.get('current'):
        meas_idx['Cm_idx'] = np.ones(len(branch.i)).astype(bool)

    meas_types = ['Pf', 'Qf', 'Cm', 'Pi', 'Qi', 'Vm']
    agg_meas_idx = aggregate_meas_idx(meas_idx , meas_types)

    h_ac_cart = H_AC_cartesian(sys, branch, meas_idx)
    h_ac_polar = H_AC_polar(sys, branch, meas_idx)
    z, _ = h_ac_polar.estimate(V=V_true, T=T_true)
    var = np.ones(len(z))
    if kwargs.get('noise'):
        var = [
            np.repeat(kwargs.get(f'{meas_type}_noise', 1),
                      len(agg_meas_idx[meas_type]))
            for meas_type in meas_types
        ]
        var = np.concatenate(var)
        noise = np.random.normal(np.zeros_like(z), np.sqrt(var))
        z += noise

    return z, var, meas_idx, agg_meas_idx, h_ac_cart, h_ac_polar, T_true, V_true, Vc_true


def square_mag(z, var, agg_meas_idx):
    z_square = z.copy()
    mag_idx = np.r_[agg_meas_idx['Cm'], agg_meas_idx['Vm']]
    z_square[mag_idx] = z[mag_idx] ** 2
    var_square = var.copy()
    var_square[mag_idx] = 2 * (var[mag_idx] ** 2)
    return z_square, var_square


def calc_dT(T_true, T_est):
    T_true_deg = np.rad2deg(T_true)
    T_est_deg = np.rad2deg(T_est)
    return np.deg2rad((T_true_deg - T_est_deg + 180) % 360 - 180)
    # return T_true - T_est

def RMSE(T_true, V_true, T_est, V_est):
    u_est = V_est * np.exp(1j * T_est)
    u_true = V_true * np.exp(1j * T_true)
    err = np.linalg.norm(u_est - u_true) / np.linalg.norm(u_true)
    return np.real(err)

def iterative_err(T_true, V_true, T_est=None, V_est=None, **kwargs):
    u_est = kwargs.get('u_est', None)
    if u_est is None:
        u_est = V_est * np.exp(1j * T_est)
    u_true = V_true * np.exp(1j * T_true)

    UUH_true = np.outer(u_true, np.conj(u_true))
    UUH_est = np.outer(u_est, np.conj(u_est))
    return np.linalg.norm(UUH_true - UUH_est, ord='fro') / np.linalg.norm(UUH_true, ord='fro')


def sample_from_SDR(Vc_est, V_opt, sys, T_true, V_true, num_samples=50):
    V_real, V_imag = np.real(Vc_est), np.imag(Vc_est)

    T = np.arctan2(V_imag, V_real)
    V = np.sqrt(V_real ** 2 + V_imag ** 2)

    best_fit = RMSE(T_true, V_true, T, V, sys.nb)

    T_best = T
    V_best = V

    for _ in range(num_samples):
        mu = np.zeros(sys.nb * 2)
        cov = .5 * np.block([[np.real(V_opt), -np.imag(V_opt)],
                             [np.imag(V_opt), np.real(V_opt)]])
        Vc = np.random.multivariate_normal(mu, cov)
        Vc = Vc[:sys.nb] + 1j * Vc[sys.nb:]
        V_real, V_imag = np.real(Vc), np.imag(Vc)
        T = np.arctan2(V_imag, V_real)
        V = np.sqrt(V_real ** 2 + V_imag ** 2)

        fit = RMSE(T_true, V_true, T, V, sys.nb)

        if fit < best_fit:
            best_fit = fit
            T_best = T
            V_best = V

    return T_best, V_best


def sample_from_SGD(u, H, measurements, num_samples=0):
    n = u.shape[0]
    UUH = np.outer(u, np.conj(u))
    if u.size > u.shape[0]:
        eigvals, eigvecs = np.linalg.eigh(UUH)
        idx_max = np.argmax(eigvals)
        u = np.sqrt(eigvals[idx_max]) * eigvecs[:, idx_max]

    err = SGD_se_obj(u, H, measurements)

    for _ in range(num_samples):
        mu = np.zeros(n * 2)
        cov = .5 * np.block([[np.real(UUH), -np.imag(UUH)],
                             [np.imag(UUH), np.real(UUH)]])
        u_curr = np.random.multivariate_normal(mu, cov)
        u_curr = u_curr[:n] + 1j * u_curr[n:]
        # u_curr = u_curr.reshape(-1, 1)
        err_curr = SGD_se_obj(u_curr, H, measurements)
        if err_curr < err:
            u = u_curr
            err = err_curr

    T = np.angle(u).flatten()
    V = np.abs(u).flatten()

    return u, T, V, err