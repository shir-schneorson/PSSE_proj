import torch


def load_flow(data, sample=False, exact=False):
    flow_meas = data.get('flow')

    if flow_meas is not None:
        Pf_idx = torch.tensor(flow_meas[:, 4]).bool()
        if sample:
            Pf_idx &= torch.randint(2, (len(flow_meas),)).bool()
        if ~torch.any(Pf_idx):
            # Pf_idx = torch.randint(2, (len(flow_meas),)).bool()
            Pf_idx = torch.ones(len(flow_meas), dtype=torch.bool)

        if exact:
            Pf_meas = torch.tensor(flow_meas[Pf_idx][:, 8])
            # Pf_meas_var = flow_meas[Pf_idx][:, 3]
            Pf_meas_var = torch.ones_like(Pf_meas)

        else:
            Pf_meas = torch.tensor(flow_meas[Pf_idx][:, 2])
            Pf_meas_var = torch.tensor(flow_meas[Pf_idx][:, 3])

        Qf_idx = torch.tensor(flow_meas[:, 7]).bool()
        if sample:
            Qf_idx &= torch.randint(2, (len(flow_meas),)).bool()
        if ~torch.any(Qf_idx):
            # Qf_idx = torch.randint(2, (len(flow_meas),)).bool()
            Qf_idx = torch.ones(len(flow_meas), dtype=torch.bool)

        if exact:
            Qf_meas = torch.tensor(flow_meas[Qf_idx][:, 9])
            # Qf_meas_var = flow_meas[Qf_idx][:, 6]
            Qf_meas_var = torch.ones_like(Qf_meas)
        else:
            Qf_meas = torch.tensor(flow_meas[Qf_idx][:, 5])
            Qf_meas_var = torch.tensor(flow_meas[Qf_idx][:, 6])

        zf = torch.cat([Pf_meas, Qf_meas])
        var = torch.cat([Pf_meas_var, Qf_meas_var])

        return zf, var, Pf_idx, Qf_idx

    return torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])


def load_current(data, sample=False, exact=False):
    current_meas = data.get('current')

    if current_meas is not None:
        Cm_idx = torch.tensor(current_meas[:, 4]).bool()
        if sample:
            Cm_idx &= torch.randint(2, (len(current_meas),)).bool()
        if ~torch.any(Cm_idx):
            # Cm_idx = torch.randint(2, (len(current_meas),)).bool()
            Cm_idx = torch.ones(len(current_meas), dtype=torch.bool)

        if exact:
            Cm_meas = torch.tensor(current_meas[Cm_idx][:, 5])
            # Cm_meas_var = current_meas[Cm_idx][:, 3]
            Cm_meas_var = torch.ones_like(Cm_meas)
        else:
            Cm_meas = torch.tensor(current_meas[Cm_idx][:, 2])
            Cm_meas_var = torch.tensor(current_meas[Cm_idx][:, 3])

        zc = Cm_meas
        var = Cm_meas_var

        return zc, var, Cm_idx

    return torch.tensor([]), torch.tensor([]), torch.tensor([])


def load_injections(data, sample=False, exact=False):
    injection_meas = data.get('injection')

    if injection_meas is not None:
        Pi_idx = torch.tensor(injection_meas[:, 3]).bool()
        if sample:
            Pi_idx &= torch.randint(2, (len(injection_meas),)).bool()
        if ~torch.any(Pi_idx):
            # Pi_idx = torch.randint(2, (len(injection_meas),)).bool()
            Pi_idx = torch.ones(len(injection_meas), dtype=torch.bool)

        if exact:
            Pi_meas = torch.tensor(injection_meas[Pi_idx][:, 7])
            # Pi_meas_var = injection_meas[Pi_idx][:, 2]
            Pi_meas_var = torch.ones_like(Pi_meas)
        else:
            Pi_meas = torch.tensor(injection_meas[Pi_idx][:, 1])
            Pi_meas_var = torch.tensor(injection_meas[Pi_idx][:, 2])

        Qi_idx = torch.tensor(injection_meas[:, 6]).bool()
        if sample:
            Qi_idx &= torch.randint(2, (len(injection_meas),)).bool()
        if ~torch.any(Qi_idx):
            # Qi_idx = torch.randint(2, (len(injection_meas),)).bool()
            Qi_idx = torch.ones(len(injection_meas), dtype=torch.bool)
        if exact:
            Qi_meas = torch.tensor(injection_meas[Qi_idx][:, 8])
            # Qi_meas_var = injection_meas[Qi_idx][:, 5]
            Qi_meas_var = torch.ones_like(Qi_meas)
        else:
            Qi_meas = torch.tensor(injection_meas[Qi_idx][:, 4])
            Qi_meas_var = torch.tensor(injection_meas[Qi_idx][:, 5])

        zi = torch.cat([Pi_meas, Qi_meas])
        var = torch.cat([Pi_meas_var, Qi_meas_var])

        return zi, var, Pi_idx, Qi_idx

    return torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])


def load_voltage(data, sample=False, exact=False):
    voltage_meas = data.get('voltage')

    if voltage_meas is not None:
        Vm_idx = torch.tensor(voltage_meas[:, 3]).bool()
        if sample:
            Vm_idx &= torch.randint(2, (len(voltage_meas),)).bool()
        if ~torch.any(Vm_idx):
            Vm_idx = torch.randint(2, (len(voltage_meas),)).bool()
            Vm_idx = torch.ones(len(voltage_meas), dtype=torch.bool)
        if exact:
            Vm_meas = torch.tensor(voltage_meas[Vm_idx][:, 4])
            # Vm_meas_var = voltage_meas[Vm_idx][:, 2]
            Vm_meas_var = torch.ones_like(Vm_meas)
        else:
            Vm_meas = torch.tensor(voltage_meas[Vm_idx][:, 1])
            Vm_meas_var = torch.tensor(voltage_meas[Vm_idx][:, 2])

        zv = Vm_meas
        var = Vm_meas_var

        return zv, var, Vm_idx

    return torch.tensor([]), torch.tensor([]), torch.tensor([])


def load_legacy_measurements(legacy_data, **kwargs):
    sample = kwargs.get('sample', False)
    exact = kwargs.get('exact', False)
    types = ['flow', 'current', 'injection', 'voltage']
    measurement_type = kwargs.get('measurement_type', types)
    z = torch.tensor([])
    v = torch.tensor([])
    indices = {}
    if 'flow' in measurement_type:
        zf, varf, Pf_idx, Qf_idx = load_flow(legacy_data, sample, exact)
        z = torch.cat([z, zf]) if z.nelement() > 0 else zf
        v = torch.cat([v, varf]) if v.nelement() > 0 else varf
        indices['Pf_idx'] = Pf_idx
        indices['Qf_idx'] = Qf_idx

    if 'current' in measurement_type:
        zc, varc, Cm_idx = load_current(legacy_data, sample, exact)
        z = torch.cat([z, zc]) if z.nelement() > 0 else zc
        v = torch.cat([v, varc]) if v.nelement() > 0 else varc
        indices['Cm_idx'] = Cm_idx

    if 'injection' in measurement_type:
        zi, vari, Pi_idx, Qi_idx = load_injections(legacy_data, sample, exact)
        z = torch.cat([z, zi]) if z.nelement() > 0 else zi
        v = torch.cat([v, vari]) if v.nelement() > 0 else vari
        indices['Pi_idx'] = Pi_idx
        indices['Qi_idx'] = Qi_idx

    if 'voltage' in measurement_type:
        zv, varv, Vm_idx = load_voltage(legacy_data, sample, exact)
        z = torch.cat([z, zv]) if z.nelement() > 0 else zv
        v = torch.cat([v, varv]) if v.nelement() > 0 else varv
        indices['Vm_idx'] = Vm_idx

    return z.float(), v.float(), indices
