import numpy as np


def init_start_point(sys, data, how='flat',
                     flat_init=(0, 1), random_init=(-0.5, 0.5, 0.95, 1.05)):
    if how == 'flat':
        T = np.deg2rad(np.repeat(flat_init[0], sys.nb))
        T[sys.slk_bus[0]] = sys.slk_bus[1]
        V = np.repeat(flat_init[1], sys.nb)

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

    else:
        deg_min, deg_max = sorted(np.deg2rad(random_init[:2]))
        T = (deg_max - deg_min) * np.random.rand(sys.nb) + deg_min
        T[sys.slk_bus[0]] = sys.slk_bus[1]

        vm_min, vm_max = sorted(random_init[2:])
        V = (vm_max - vm_min) * np.random.rand(sys.nb) + vm_min

    return T, V
