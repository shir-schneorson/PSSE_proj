import json
import os
import re

import numpy as np
import pandas as pd
from tqdm import tqdm

from init_net.process_net_data import parse_ieee_mat, System
from power_flow_ac.NR_pf import NR_PF


def flatten_series(entry):
    series_id = entry['series_id']
    rows = np.c_[entry['data']]
    times = pd.to_datetime(rows[:, 0], format="%Y%m%dT%H")
    values = rows[:, 1].astype(float)
    df = pd.DataFrame({'timestamp': times, series_id: values})
    return df.set_index("timestamp")


def create_gen_demand_df():
    try:
        full_df = pd.read_csv('data/EBA_filtered_df.csv', index_col=0)
        gen_df = pd.read_csv('data/EBA_gen_df.csv', index_col=0)
        demand_df = pd.read_csv('data/EBA_demand_df.csv', index_col=0)

    except FileNotFoundError:
        filtered_data = json.load(open('./data/EBA_filtered.json'))
        all_dfs = [flatten_series(entry) for entry in filtered_data]  # From previous steps
        full_df = pd.concat(all_dfs, axis=1)
        demand_cols = [c for c in full_df.columns if re.match(r"EBA\..*\.D\.H", c)]
        gen_cols = [c for c in full_df.columns if re.match(r"EBA\..*\.NG\.H", c)]

        gen_df = full_df[gen_cols].fillna(0)
        demand_df = full_df[demand_cols].fillna(0)

    return gen_df, demand_df, full_df

def sample_gen_load_reg(demand_df, gen_df, sys):
    gen_idx = np.where(sys.bus.bus_type == 2)[0]
    regions = [col.split('.')[1].split('-')[0] for col in demand_df.columns if 'ALL' in col]
    sub_regions = {reg: np.array([col for col in demand_df.columns if reg in col and '-ALL' not in col]) for reg in regions}
    chosen_regs = {}
    count = 0
    rem_regs = list(sub_regions.items())
    while count < 118:
        reg, sub_regs = rem_regs.pop()
        if len(sub_regs) > 0:
            buses = np.arange(count, count + len(sub_regs))
            gen_regs = [col for col in gen_df.columns if f'{reg}-ALL.NG.H' in col]
            gen_buses = gen_idx[np.isin(gen_idx, buses)]
            chosen_regs[reg] = {'sub_regs': np.array(sub_regs),
                                'buses': buses,
                                'gen_regs': gen_regs,
                                'gen_buses': gen_buses}
            count += len(sub_regs)

    return chosen_regs

def generate_power(gen_df, demand_df, time, sys, regions):
    sys = sys.copy()
    bus_df = sys.bus.copy()
    gen_idx = np.where(bus_df.bus_type == 2)[0]
    for reg_id, reg in regions.items():
        bus_df.loc[reg['buses'], 'Pl'] = demand_df.loc[time, reg['sub_regs']].values
        num_gens = len(reg['gen_buses'])
        bus_df.loc[reg['gen_buses'], 'Pg'] = gen_df.loc[time, reg['gen_regs']].iloc[0] / max(num_gens, 1)

    rng = np.random.default_rng()
    a = 25
    pf = rng.power(a, len(bus_df))
    bus_df.loc[:, 'Ql'] = bus_df.loc[:, 'Pl'].values * np.tan(np.arccos(pf))

    sum_Pg = np.sum(bus_df.loc[gen_idx, 'Pg'].values)
    sum_Sl = np.sum(np.abs(bus_df.loc[:, 'Pl'].values + 1j * bus_df.loc[:, 'Ql'].values))
    if sum_Pg == 0 or sum_Sl == 0:
        raise ValueError("Bad timestamp")
    bus_df.loc[gen_idx, 'Pg'] = bus_df.loc[gen_idx, 'Pg'].values * 10 / sum_Pg
    bus_df.loc[:, ['Pl', 'Ql']] = bus_df.loc[:, ['Pl', 'Ql']].values * 10 / sum_Sl

    x_init = bus_df.loc[:, ['To', 'Vo']].values
    loads = bus_df.loc[:, ['Pl', 'Ql']].values
    gens = bus_df.loc[:, ['Pg', 'Qg']].values

    user = {'list': ['voltage'], 'stop': 1e-8, 'maxIter': 500}

    pf = NR_PF(sys, loads, gens, x_init, user)

    V, T = np.abs(pf['Vc']), np.angle(pf['Vc'])
    Pl, Ql = loads.T
    Pg, Qg = gens.T

    return V, T, Pl, Ql, Pg, Qg


def create_data_source(timestamps, gen_df, demand_df, sys):
    regions = sample_gen_load_reg(demand_df, gen_df, sys)
    V_all, T_all, Pl_all, Ql_all, Pg_all, Qg_all = [], [], [], [], [], []
    timestamps_all = []
    for timestamp in tqdm(timestamps, desc='Generating data'):
        try:
            V, T, Pl, Ql, Pg, Qg = generate_power(gen_df, demand_df, timestamp, sys, regions)
            V_all.append(V)
            T_all.append(T)
            Pl_all.append(Pl)
            Ql_all.append(Ql)
            Pg_all.append(Pg)
            Qg_all.append(Qg)
            timestamps_all.append(timestamp)
        except ValueError:
            continue

    Vm = pd.DataFrame(np.r_[V_all],
                      index=np.r_[timestamps_all],
                      columns=[f'V{i}' for i in range(len(V_all[0]))])
    Va = pd.DataFrame(np.r_[T_all],
                      index=np.r_[timestamps_all],
                      columns=[f'T{i}' for i in range(len(T_all[0]))])
    Pl = pd.DataFrame(np.r_[Pl_all],
                      index=np.r_[timestamps_all],
                      columns=[f'Pl{i}' for i in range(len(Pl_all[0]))])
    Ql = pd.DataFrame(np.r_[Ql_all],
                      index=np.r_[timestamps_all],
                      columns=[f'Ql{i}' for i in range(len(Ql_all[0]))])
    Pg = pd.DataFrame(np.r_[Pg_all],
                      index=np.r_[timestamps_all],
                      columns=[f'Pg{i}' for i in range(len(Pg_all[0]))])
    Qg = pd.DataFrame(np.r_[Qg_all],
                      index=np.r_[timestamps_all],
                      columns=[f'Qg{i}' for i in range(len(Qg_all[0]))])

    return Vm, Va, Pl, Ql, Pg, Qg


def main():
    gen_df, demand_df, full_df = create_gen_demand_df()

    file = '../nets/ieee118_186.mat'

    data = parse_ieee_mat(file)
    system_data = data['data']['system']
    sys = System(system_data)

    n_samples = 20000
    timestamps = np.random.choice(gen_df.index, n_samples, replace=False)
    Vm, Va, Pl, Ql, Pg, Qg = create_data_source(timestamps, gen_df, demand_df, sys)
    output_dir = "../data_parser/data/time_series2"
    Vm.to_csv(os.path.join(output_dir, 'ieee118_186_Vm_timeseries.csv'))
    Va.to_csv(os.path.join(output_dir, 'ieee118_186_Va_timeseries.csv'))
    Pl.to_csv(os.path.join(output_dir, 'ieee118_186_Pl_timeseries.csv'))
    Ql.to_csv(os.path.join(output_dir, 'ieee118_186_Ql_timeseries.csv'))
    Pg.to_csv(os.path.join(output_dir, 'ieee118_186_Pg_timeseries.csv'))
    Qg.to_csv(os.path.join(output_dir, 'ieee118_186_Qg_timeseries.csv'))


if __name__ == "__main__":
    main()
