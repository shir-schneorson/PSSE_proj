import json
import os
import re
import tempfile

import numpy as np
import pandas as pd

import pandapower as pp
from pandapower.control import ConstControl
from pandapower.networks import case118
from pandapower.timeseries import OutputWriter, DFData, run_timeseries


def flatten_series(entry):
    series_id = entry['series_id']
    rows = np.c_[entry['data']]
    times = pd.to_datetime(rows[:, 0], format="%Y%m%dT%H")
    values =  rows[:, 1].astype(float)
    df = pd.DataFrame({'timestamp': times, series_id: values})
    return df.set_index("timestamp")


def create_gen_demand_df():
    try:
        full_df = pd.read_csv('data/EBA_filtered_df.csv')
        gen_df = pd.read_csv('data/EBA_gen_df.csv')
        demand_df = pd.read_csv('data/EBA_demand_df.csv')

        gen_df.set_index("timestamp", inplace=True)
        demand_df.set_index("timestamp", inplace=True)
        full_df.set_index("timestamp", inplace=True)

    except FileNotFoundError:
        filtered_data = json.load(open('data/EBA_filtered.json'))
        all_dfs = [flatten_series(entry) for entry in filtered_data]  # From previous steps
        full_df = pd.concat(all_dfs, axis=1)
        demand_cols = [c for c in full_df.columns if re.match(r"EBA\..*\.D\.H", c)]
        gen_cols = [c for c in full_df.columns if re.match(r"EBA\..*\.NG\.H", c)]

        gen_df = full_df[gen_cols].fillna(0)
        demand_df = full_df[demand_cols].fillna(0)

    return gen_df, demand_df, full_df



def compute_sgen_PQ(timestamp, region_groups, df, net):
    """Return P and Q for each bus by summing region group values at a timestamp"""
    rng = np.random.default_rng()
    a = 25.  # shape
    P = np.array([df.loc[timestamp, group].values.sum() for group in region_groups])
    scale = compute_global_scale(P, net.sgen.max_p_mw.values, net.sgen.min_p_mw.values)
    P *= scale

    pf = rng.power(a, len(P))
    Q = P * np.tan(np.arccos(pf))
    scale = compute_global_scale(Q, net.sgen.max_q_mvar.values, net.sgen.min_q_mvar.values)
    Q *= scale
    return P, Q

def compute_load_PQ(timestamp, region_groups, df, net):
    """Return P and Q for each bus by summing region group values at a timestamp"""
    rng = np.random.default_rng()
    a = 25.  # shape
    P = np.array([df.loc[timestamp, group].values.sum() for group in region_groups])

    pf = rng.power(a, len(P))
    Q = P * np.tan(np.arccos(pf))

    P /= net.sn_mva
    Q /= net.sn_mva

    return P, Q


def compute_global_scale(P_values, P_max, P_min):
    s_min = float('-inf')
    s_max = float('inf')

    for p, pmin, pmax in zip(P_values, P_min, P_max):
        if p == 0:
            continue

        s1 = pmin / p
        s2 = pmax / p

        s_low, s_high = sorted([s1, s2])

        s_min = max(s_min, s_low)
        s_max = min(s_max, s_high)

    if s_min > s_max:
        raise ValueError("No common scaling factor can satisfy all constraints.")

    if s_min <= 1 <= s_max:
        return 1.0
    else:
        return min(s_max, 1.0)


def generate_power(gen_df, demand_df, time, net):
    n_load_buses = len(net.load)
    m_gen_buses = len(net.sgen)

    demand_cols = demand_df.columns
    gen_cols = gen_df.columns
    load_region_groups = np.array_split(demand_cols, n_load_buses)
    gen_region_groups = np.array_split(gen_cols, m_gen_buses)

    p_max, p_min = net.sgen.max_p_mw.values, net.sgen.min_p_mw.values
    q_max, q_min = net.sgen.max_q_mvar.values, net.sgen.min_q_mvar.values
    P_loads, Q_loads = compute_load_PQ(time, load_region_groups, demand_df, net)
    P_gens, Q_gens = compute_sgen_PQ(time, gen_region_groups, gen_df, net)

    # P_loads /= net.sn_mva
    # Q_loads /= net.sn_mva
    #
    # P_gens /= net.sn_mva
    # Q_gens /= net.sn_mva
    # scale = compute_global_scale(P_gens, net.sgen.max_p_mw.values, net.sgen.min_p_mw.values)
    #
    # P_gens *= scale
    # Q_gens *= scale
    # P_loads *= scale
    # Q_loads *= scale

    return P_loads, Q_loads, P_gens, Q_gens


def create_output_writer(net, time_steps, output_dir):
    ow = OutputWriter(net, time_steps, output_path=output_dir, output_file_type=".csv", log_variables=list())
    # these variables are saved to the harddisk after / during the time series loop
    ow.log_variable('res_load', 'p_mw')
    ow.log_variable('res_load', 'q_mvar')

    ow.log_variable('res_gen', 'p_mw')
    ow.log_variable('res_gen', 'q_mvar')

    ow.log_variable('res_bus', 'vm_pu')
    ow.log_variable('res_bus', 'va_degree')
    ow.log_variable('res_bus', 'p_mw')
    ow.log_variable('res_bus', 'q_mvar')

    ow.log_variable('res_line', 'p_from_mw')
    ow.log_variable('res_line', 'q_from_mvar')
    ow.log_variable('res_line', 'p_to_mw')
    ow.log_variable('res_line', 'q_to_mvar')
    return ow


def create_controllers(net, ds):
    load_p_names = [f'load{i}_p' for i in net.load.bus.values]
    load_q_names = [f'load{i}_q' for i in net.load.bus.values]
    sgen_p_names = [f'sgen{i}_p' for i in net.sgen.bus.values]
    sgen_q_names = [f'sgen{i}_q' for i in net.sgen.bus.values]

    ConstControl(net, element='load', variable='p_mw', element_index=net.load.index,
                 data_source=ds, profile_name=load_p_names)
    ConstControl(net, element='load', variable='q_mvar', element_index=net.load.index,
                 data_source=ds, profile_name=load_q_names)
    ConstControl(net, element='sgen', variable='p_mw', element_index=net.sgen.index,
                 data_source=ds, profile_name=sgen_p_names)
    ConstControl(net, element='sgen', variable='q_mvar', element_index=net.sgen.index,
                 data_source=ds, profile_name=sgen_q_names)

def create_data_source(timestamps, gen_df, demand_df, net):
    load_p = []
    load_q = []
    sgen_p = []
    sgen_q = []
    for timestamp in timestamps:
        P_loads, Q_loads, P_gens, Q_gens = generate_power(gen_df, demand_df, timestamp, net)
        load_p.append(P_loads)
        load_q.append(Q_loads)
        sgen_p.append(P_gens)
        sgen_q.append(Q_gens)

    profiles = pd.DataFrame()
    # profiles['timestamp'] = timestamps
    load_p_names = [f'load{i}_p' for i in net.load.bus.values]
    load_q_names = [f'load{i}_q' for i in net.load.bus.values]
    sgen_p_names = [f'sgen{i}_p' for i in net.sgen.bus.values]
    sgen_q_names = [f'sgen{i}_q' for i in net.sgen.bus.values]

    profiles[load_p_names] = np.r_[load_p]
    profiles[load_q_names] = np.r_[load_q]
    profiles[sgen_p_names] = np.r_[sgen_p]
    profiles[sgen_q_names] = np.r_[sgen_q]
    # profiles.set_index('timestamp', inplace=True)

    ds = DFData(profiles)

    return profiles, ds, load_p_names, load_q_names, sgen_p_names, sgen_q_names


def convert_gen_to_sgen(net, drop_original=True):
    for _, gen_row in net.gen.iterrows():
        pp.create_sgen(
            net,
            bus=gen_row["bus"],
            p_mw=gen_row["p_mw"],
            max_p_mw=gen_row["max_p_mw"],
            min_p_mw=gen_row["min_p_mw"],
            max_q_mvar=gen_row["max_q_mvar"],
            min_q_mvar=gen_row["min_q_mvar"],
        )
    # if drop_original:
    #     net.gen.drop(index=net.gen.index, inplace=True)


def main():
    gen_df, demand_df, full_df = create_gen_demand_df()
    net = case118()
    convert_gen_to_sgen(net)
    n_samples = 10
    timestamps = np.random.choice(gen_df.index, n_samples, replace=False)
    timestamps_index = np.arange(len(timestamps))
    profiles, ds, load_p_names, load_q_names, sgen_p_names, sgen_q_names = create_data_source(timestamps, gen_df, demand_df, net)
    output_dir = "data/time_series"
    create_output_writer(net, timestamps_index, output_dir)
    create_controllers(net, ds)

    converged = False
    scale = .95
    while not converged:
        try:
            create_controllers(net, ds)
            run_timeseries(net, timestamps_index)
            converged = True
        except pp.LoadflowNotConverged:
            profiles[load_p_names] *= scale
            profiles[load_q_names] *= scale
            ds = DFData(profiles)
            net.controller.drop(index=net.controller.index, inplace=True)

if __name__ == "__main__":
    main()