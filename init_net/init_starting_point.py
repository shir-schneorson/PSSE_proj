import numpy as np

from power_flow_dc.solve_dcpf import generate_Ybus_dc, H_DC, DC_PF


def init_start_point(sys, data, how='flat', dc_init=None,
                     flat_init=(0, 1), random_init=(0.3, 1, 1e-2)):
    if how == 'flat':
        T = np.deg2rad(np.repeat(flat_init[0], sys.nb))
        T[sys.slk_bus[0]] = sys.slk_bus[1]
        V = np.repeat(flat_init[1], sys.nb)
        V[sys.slk_bus[0]] = sys.slk_bus[2]

    elif how == 'exact':
        pmu = data['data'].get('pmu')
        if pmu is None:
            T = sys.bus['To']
            V = sys.bus['Vo']
        else:
            T = pmu['voltage'][:, -1]
            V = pmu['voltage'][:, -2]

    elif how == 'warm':
        T = sys.bus['To']
        V = sys.bus['Vo']

    elif how == 'dc':
        branch, z, v, meas_idx, agg_meas_idx = dc_init

        Ybus, sys.branch_data, sys.bus_data = generate_Ybus_dc(sys.bus, sys.branch)
        h_ac = H_DC(sys, branch, Ybus, meas_idx, agg_meas_idx, z, v)

        T = DC_PF(sys.slk_bus, h_ac)
        V = np.repeat(flat_init[1], sys.nb)

    elif how == 'random':
        theta = np.pi * random_init[0]
        # T = np.random.uniform(-theta, theta, sys.nb)
        T = np.random.normal(0, theta / 3, sys.nb)
        T[sys.slk_bus[0]] = 0

        # mu, sigma = random_init[1], np.sqrt(random_init[2])
        # V = np.random.normal(mu, sigma, sys.nb)
        V_min, V_max = random_init[1], np.sqrt(random_init[2])
        # V = np.random.uniform(V_min, V_max, sys.nb)
        V = np.random.normal(1, 0.05 / 3, sys.nb)
        V[sys.slk_bus[0]] = 1

    else:
        T = np.random.rand(sys.nb) -.5
        T[sys.slk_bus[0]] = sys.slk_bus[1]

        V = (1.05 - .95) * np.random.rand(sys.nb) + .95

    return T, V
