from init_net.process_net_data import parse_ieee_mat, System
from power_flow_ac.NR_pf import NR_PF

file = '../nets/ieee118_186.mat'

data = parse_ieee_mat(file)
system_data = data['data']['system']
sys = System(system_data)

x_init = sys.bus.loc[:, ['To', 'Vo']].values
loads = sys.bus.loc[:, ['Pl', 'Ql']].values
gens = sys.bus.loc[:, ['Pg', 'Qg']].values
Pl, Ql = sys.bus.Pl.values, sys.bus.Ql.values
Pg, Qg = sys.bus.Pg.values, sys.bus.Qg.values

user = {'list': ['voltage'], 'stop': 1e-8, 'maxIter': 500}

pf = NR_PF(sys, loads, gens, x_init, user)

print()