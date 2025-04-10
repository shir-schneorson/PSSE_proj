import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.sparse import coo_array

def parse_ieee_mat(file):
    data = loadmat(file)  # Ensure this file contains 'data.system.line'
    data['data'] = {n: data['data'][n][0, 0] for n in data['data'].dtype.names}
    data['data']['system'] = {n: data['data']['system'][n][0, 0] for n in data['data']['system'].dtype.names}
    data['data']['legacy'] = {n: data['data']['legacy'][n][0, 0] for n in data['data']['legacy'].dtype.names}
    data['data']['pmu'] = {n: data['data']['pmu'][n][0, 0] for n in data['data']['pmu'].dtype.names}
    return data


def process_branch_data(line_data, in_transformer_data, shift_transformer_data):
    cols = ['idx_from', 'idx_to', 'rij', 'xij', 'bsi', 'tij', 'fij']

    line_data = line_data[line_data[:, -1] == 1, :-1]
    nbl = len(line_data)
    tij = np.ones(nbl)
    fij = np.zeros(nbl)

    branch_data = np.c_[line_data, tij, fij]

    if in_transformer_data is not None:
        in_transformer_data = in_transformer_data[in_transformer_data[:, -1] == 1, :-1]
        nbit = len(in_transformer_data)
        fij = np.zeros(nbit)
        in_transformer_data = np.c_[in_transformer_data, fij]
        branch_data = np.r_[branch_data, in_transformer_data]
    if shift_transformer_data is not None:
        branch_data = np.r_[branch_data, shift_transformer_data]

    branch_data[:, [0, 1]] -= 1
    branch_data[:, -1] = np.deg2rad(branch_data[:, -1])
    branch_data_df = pd.DataFrame(branch_data, columns=cols)

    return branch_data_df


def process_bus_data(bus_data, generator_data, base_MVA):
    cols = ['idx_bus', 'bus_type', 'Vo', 'To', 'Pl', 'Ql', 'rsh', 'xsh', 'Vmin', 'Vmax']

    nb = len(bus_data)
    bus_data[:, 0] -= 1
    bus_data[:, 2] = np.deg2rad(bus_data[:, 2])
    bus_data[:, 4:8] /= base_MVA
    bus_data[:, 3] = np.deg2rad(bus_data[:, 3])
    bus_data_df = pd.DataFrame(bus_data, columns=cols)
    bus_data_df.sort_values(by=['idx_bus'], inplace=True, ignore_index=True)

    slack_bus = bus_data_df.loc[bus_data_df['bus_type'] == 3, 'idx_bus'].iloc[0].astype(int)
    slack_bus = (slack_bus, bus_data_df.loc[slack_bus, 'To'])

    gen_cols =['Pg', 'Qg', 'Qmin', 'Qmax']
    if generator_data is not None:
        generator_data = generator_data[generator_data[:, -1] == 1, :-1]
        idx_bus_gen = (generator_data[:, 0] - 1).astype(int)
        generator_data[:, 1:5] /= base_MVA

        nge = len(generator_data)
        A = coo_array((np.ones(nge), [idx_bus_gen, np.arange(nge)]), (nb, nge)).toarray()
        bus_data_df.loc[:, gen_cols] = A @ generator_data[:, 1:5]
        bus_data_df.loc[idx_bus_gen, 'Vo'] = generator_data[:, -1]
    else:
        bus_data_df.loc[:, gen_cols] = 0

    return bus_data_df, slack_bus


def generate_Ybus(bus_data, branch_data):
    yij = 1 / (branch_data['rij'].values + 1j * branch_data['xij'].values)
    ysi = 1j * branch_data['bsi'].values / 2
    aij = branch_data['tij'].values * np.exp(1j * branch_data['fij'].values)

    yij_p_bsi = yij + ysi
    yij_p_bsi_d_aij_p2 = yij_p_bsi / (np.conj(aij) * aij)
    m_yij_d_conj_aij = -yij / np.conj(aij)
    m_yij_d_aij = -yij / aij

    nb = len(bus_data)
    nbr = len(branch_data)

    idx_branch = np.array(branch_data.index)
    idx_from, idx_to = branch_data[['idx_from', 'idx_to']].values.T

    Ai = coo_array((np.ones_like(idx_branch), [idx_branch, idx_from]), (nbr, nb)).toarray()
    Yi = coo_array((yij_p_bsi_d_aij_p2, [idx_branch, idx_from]), (nbr, nb)).toarray()
    Yi += coo_array((m_yij_d_conj_aij, [idx_branch, idx_to]), (nbr, nb)).toarray()

    Aj = coo_array((np.ones_like(idx_branch), [idx_branch, idx_to]), (nbr, nb)).toarray()
    Yj = coo_array((m_yij_d_aij, [idx_branch, idx_from]), (nbr, nb)).toarray()
    Yj += coo_array((yij_p_bsi, [idx_branch, idx_to]), (nbr, nb)).toarray()

    ysh = bus_data['rsh'].values + 1j * bus_data['xsh'].values
    Ysh = np.diag(ysh)

    Ybus = (Ai.T @ Yi) + (Aj.T @ Yj) + Ysh

    branch_data.loc[:, 'yij'] = yij
    branch_data.loc[:, 'ysi'] = ysi
    branch_data.loc[:, 'aij'] = aij
    branch_data.loc[:, 'yij_p_bsi'] = yij_p_bsi
    branch_data.loc[:, 'yij_p_bsi_d_aij_p2'] = yij_p_bsi_d_aij_p2
    branch_data.loc[:, 'm_yij_d_conj_aij'] = m_yij_d_conj_aij
    branch_data.loc[:, 'm_yij_d_aij'] = m_yij_d_aij

    Yii = np.diag(np.diag(Ybus))
    Yij = Ybus - Yii


    return Ybus, Yii, Yij, branch_data

class Branch:
    def __init__(self, branch_data):
        self.no = np.arange(len(branch_data) * 2)
        self.i = np.r_[branch_data.idx_from.values, branch_data.idx_to.values]
        self.j = np.r_[branch_data.idx_to.values, branch_data.idx_from.values]
        self.gij = np.r_[np.real(branch_data.yij.values), np.real(branch_data.yij.values)]
        self.bij = np.r_[np.imag(branch_data.yij.values), np.imag(branch_data.yij.values)]
        self.bsi = np.r_[branch_data.bsi.values / 2, branch_data.bsi.values / 2]
        self.tij = np.r_[1 / branch_data.tij.values, np.ones(len(branch_data))]
        self.pij = np.r_[1 / branch_data.tij.values, 1 / branch_data.tij.values]
        self.fij = np.r_[branch_data.fij.values, -branch_data.fij.values]


class System:
    def __init__(self, system_data):
        self.baseMVA = system_data['baseMVA'][0, 0]
        bus_data, generator_data = system_data['bus'], system_data.get('generator')
        line_data = system_data['line']
        in_transformer_data = system_data.get('inTransformer')
        shift_transformer_data = system_data.get('shiftTransformer')
        self.bus, self.slk_bus = process_bus_data(bus_data, generator_data, self.baseMVA)
        self.branch = process_branch_data(line_data, in_transformer_data, shift_transformer_data)
        self.nbr = len(self.branch)
        self.nb = len(self.bus)
        self.Ybus, self.Yii, self.Yij, self.branch = generate_Ybus(self.bus, self.branch)
