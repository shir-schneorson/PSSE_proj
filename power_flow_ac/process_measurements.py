import numpy as np


def load_flow(data, sample=False):
    flow_meas = data.get('flow')

    if flow_meas is not None:
        Pf_idx = flow_meas[:, 4].astype(bool)
        if sample:
            Pf_idx &= np.random.randint(2, size=len(flow_meas)).astype(bool)

        Pf_meas = flow_meas[Pf_idx][:, 2]
        Pf_meas_var = flow_meas[Pf_idx][:, 3]

        Qf_idx = flow_meas[:, 7].astype(bool)
        if sample:
            Qf_idx &= np.random.randint(2, size=len(flow_meas)).astype(bool)

        Qf_meas = flow_meas[Qf_idx][:, 5]
        Qf_meas_var = flow_meas[Qf_idx][:, 6]

        zf = np.r_[Pf_meas, Qf_meas]
        var = np.r_[Pf_meas_var, Qf_meas_var]

        return zf, var, Pf_idx, Qf_idx

    return np.array([]), np.array([]), np.array([]), np.array([])


def load_current(data, sample=False):
    current_meas = data.get('current')

    if current_meas is not None:
        Cm_idx = current_meas[:, 4].astype(bool)
        if sample:
            Cm_idx &= np.random.randint(2, size=len(current_meas)).astype(bool)

        Cm_meas = current_meas[Cm_idx][:, 2]
        Cm_meas_var = current_meas[Cm_idx][:, 3]

        zf = Cm_meas
        var = Cm_meas_var

        return zf, var, Cm_idx

    return np.array([]), np.array([]), np.array([])


def load_injections(data, sample=False):
    injection_meas = data.get('injection')

    if injection_meas is not None:
        Pi_idx = injection_meas[:, 3].astype(bool)
        if sample:
            Pi_idx &= np.random.randint(2, size=len(injection_meas)).astype(bool)
        Pi_meas = injection_meas[Pi_idx][:, 1]
        Pi_meas_var = injection_meas[Pi_idx][:, 2]

        Qi_idx = injection_meas[:, 6].astype(bool)
        if sample:
            Qi_idx &= np.random.randint(2, size=len(injection_meas)).astype(bool)
        Qi_meas = injection_meas[Qi_idx][:, 4]
        Qi_meas_var = injection_meas[Qi_idx][:, 5]

        zf = np.r_[Pi_meas, Qi_meas]
        var = np.r_[Pi_meas_var, Qi_meas_var]

        return zf, var, Pi_idx, Qi_idx

    return np.array([]), np.array([]), np.array([]), np.array([])


def load_voltage(data, sample=False):
    voltage_meas = data.get('voltage')

    if voltage_meas is not None:
        Vm_idx = voltage_meas[:, 3].astype(bool)
        if sample:
            Vm_idx &= np.random.randint(2, size=len(voltage_meas)).astype(bool)
        Vm_meas = voltage_meas[Vm_idx][:, 1]
        Vm_meas_var = voltage_meas[Vm_idx][:, 2]

        zf = Vm_meas
        var = Vm_meas_var

        return zf, var, Vm_idx

    return np.array([]), np.array([]), np.array([])


def load_legacy_measurements(legacy_data, sample=False):
    zf, varf, Pf_idx, Qf_idx = load_flow(legacy_data, sample)
    zi, vari, Pi_idx, Qi_idx = load_injections(legacy_data, sample)
    zc, varc, Cm_idx = load_current(legacy_data, sample)
    zv, varv, Vm_idx = load_voltage(legacy_data, sample)
    z = np.r_[zf, zc, zi, zv]
    v = np.r_[varf, varc, vari, varv]

    return z, v, (Pf_idx, Qf_idx, Cm_idx, Pi_idx, Qi_idx, Vm_idx)