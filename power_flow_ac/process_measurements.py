import numpy as np


def load_flow(data, sample=False, exact=False):
    flow_meas = data.get('flow')

    if flow_meas is not None:
        Pf_idx = flow_meas[:, 4].astype(bool)
        if sample:
            Pf_idx &= np.random.randint(2, size=len(flow_meas)).astype(bool)
        if ~np.all(Pf_idx):
            # Pf_idx = np.random.randint(2, size=len(flow_meas)).astype(bool)
            Pf_idx = np.ones(len(flow_meas), dtype=bool)

        if exact:
            Pf_meas = flow_meas[Pf_idx][:, 8]
            Pf_meas_var = flow_meas[Pf_idx][:, 3]
        else:
            Pf_meas = flow_meas[Pf_idx][:, 2]
            Pf_meas_var = flow_meas[Pf_idx][:, 3]


        Qf_idx = flow_meas[:, 7].astype(bool)
        if sample:
            Qf_idx &= np.random.randint(2, size=len(flow_meas)).astype(bool)
        if ~np.all(Qf_idx):
            # Qf_idx = np.random.randint(2, size=len(flow_meas)).astype(bool)
            Qf_idx = np.ones(len(flow_meas), dtype=bool)

        if exact:
            Qf_meas = flow_meas[Qf_idx][:, 9]
            Qf_meas_var = flow_meas[Qf_idx][:, 6]
        else:
            Qf_meas = flow_meas[Qf_idx][:, 5]
            Qf_meas_var = flow_meas[Qf_idx][:, 6]

        zf = np.r_[Pf_meas, Qf_meas]
        var = np.r_[Pf_meas_var, Qf_meas_var]

        return zf, var, Pf_idx, Qf_idx

    return np.array([]), np.array([]), np.array([]), np.array([])


def load_current(data, sample=False, exact=False):
    current_meas = data.get('current')

    if current_meas is not None:
        Cm_idx = current_meas[:, 4].astype(bool)
        if sample:
            Cm_idx &= np.random.randint(2, size=len(current_meas)).astype(bool)
        if ~np.all(Cm_idx):
            # Cm_idx = np.random.randint(2, size=len(current_meas)).astype(bool)
            Cm_idx = np.ones(len(current_meas), dtype=bool)

        if exact:
            Cm_meas = current_meas[Cm_idx][:, 5]
            Cm_meas_var = current_meas[Cm_idx][:, 3]
        else:
            Cm_meas = current_meas[Cm_idx][:, 2]
            Cm_meas_var = current_meas[Cm_idx][:, 3]


        zc = Cm_meas
        var = Cm_meas_var

        return zc, var, Cm_idx

    return np.array([]), np.array([]), np.array([])


def load_injections(data, sample=False, exact=False):
    injection_meas = data.get('injection')

    if injection_meas is not None:
        Pi_idx = injection_meas[:, 3].astype(bool)
        if sample:
            Pi_idx &= np.random.randint(2, size=len(injection_meas)).astype(bool)
        if ~np.all(Pi_idx):
            # Pi_idx = np.random.randint(2, size=len(injection_meas)).astype(bool)
            Pi_idx = np.ones(len(injection_meas), dtype=bool)

        if exact:
            Pi_meas = injection_meas[Pi_idx][:, 7]
            Pi_meas_var = injection_meas[Pi_idx][:, 2]
        else:
            Pi_meas = injection_meas[Pi_idx][:, 1]
            Pi_meas_var = injection_meas[Pi_idx][:, 2]

        Qi_idx = injection_meas[:, 6].astype(bool)
        if sample:
            Qi_idx &= np.random.randint(2, size=len(injection_meas)).astype(bool)
        if ~np.all(Qi_idx):
            # Qi_idx = np.random.randint(2, size=len(injection_meas)).astype(bool)
            Qi_idx = np.ones(len(injection_meas), dtype=bool)
        if exact:
            Qi_meas = injection_meas[Qi_idx][:, 8]
            Qi_meas_var = injection_meas[Qi_idx][:, 5]
        else:
            Qi_meas = injection_meas[Qi_idx][:, 4]
            Qi_meas_var = injection_meas[Qi_idx][:, 5]

        zi = np.r_[Pi_meas, Qi_meas]
        var = np.r_[Pi_meas_var, Qi_meas_var]

        return zi, var, Pi_idx, Qi_idx

    return np.array([]), np.array([]), np.array([]), np.array([])


def load_voltage(data, sample=False, exact=False):
    voltage_meas = data.get('voltage')

    if voltage_meas is not None:
        Vm_idx = voltage_meas[:, 3].astype(bool)
        if sample:
            Vm_idx &= np.random.randint(2, size=len(voltage_meas)).astype(bool)
        if ~np.all(Vm_idx):
            Vm_idx = np.random.randint(2, size=len(voltage_meas)).astype(bool)
            Vm_idx = np.ones(len(voltage_meas), dtype=bool)
        if exact:
            Vm_meas = voltage_meas[Vm_idx][:, 4]
            Vm_meas_var = voltage_meas[Vm_idx][:, 2]
        else:
            Vm_meas = voltage_meas[Vm_idx][:, 1]
            Vm_meas_var = voltage_meas[Vm_idx][:, 2]

        zv = Vm_meas
        var = Vm_meas_var

        return zv, var, Vm_idx

    return np.array([]), np.array([]), np.array([])


def load_legacy_measurements(legacy_data, **kwargs):
    sample = kwargs.get('sample', False)
    exact = kwargs.get('exact', False)
    types = ['flow', 'current', 'injection', 'voltage']
    measurement_type = kwargs.get('measurement_type', types)
    z = np.array([])
    v = np.array([])
    indices = []
    if 'flow' in measurement_type:
        zf, varf, Pf_idx, Qf_idx = load_flow(legacy_data, sample, exact)
        z = np.r_[zf]
        v = np.r_[varf]
        indices.extend([Pf_idx, Qf_idx])
    if 'current' in measurement_type:
        zc, varc, Cm_idx = load_current(legacy_data, sample, exact)
        z = np.r_[z, zc]
        v = np.r_[v, varc]
        indices.extend([Cm_idx])
    if 'injection' in measurement_type:
        zi, vari, Pi_idx, Qi_idx = load_injections(legacy_data, sample, exact)
        z = np.r_[z, zi]
        v = np.r_[v, vari]
        indices.extend([Pi_idx, Qi_idx])
    if 'voltage' in measurement_type:
        zv, varv, Vm_idx = load_voltage(legacy_data, sample, exact)
        z = np.r_[z, zv]
        v = np.r_[v, varv]
        indices.extend([Vm_idx])


    return z, v, indices