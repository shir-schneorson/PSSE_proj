import torch


def init_start_point(sys, data, how='flat',
                     flat_init=(0, 1), random_init=(0.3, 1, 1e-2)):
    if how == 'flat':
        T = torch.deg2rad(torch.full((sys.nb,), flat_init[0]))
        T[sys.slk_bus[0]] = sys.slk_bus[1]
        V = torch.full((sys.nb,), flat_init[1])

    elif how == 'exact':
        pmu = data['data'].get('pmu')
        if pmu is None:
            T = sys.bus['To']
            V = sys.bus['Vo']
        else:
            T = torch.tensor(pmu['voltage'][:, -1])
            V = torch.tensor(pmu['voltage'][:, -2])

    elif how == 'warm':
        T = sys.bus['To']
        V = sys.bus['Vo']

    elif how == 'random':
        theta = torch.pi * random_init[0]
        T = torch.empty(sys.nb).uniform_(-theta, theta)
        T[sys.slk_bus[0]] = 0

        mu, sigma = random_init[1], torch.sqrt(torch.tensor(random_init[2]))
        V = torch.normal(mu, sigma, size=(sys.nb,))
        V[sys.slk_bus[0]] = 1

    else:
        T = torch.rand(sys.nb) - 0.5
        T[sys.slk_bus[0]] = sys.slk_bus[1]

        V = (1.05 - 0.95) * torch.rand(sys.nb) + 0.95

    return T, V
