import os.path
from pathlib import Path

import torch
import torch_geometric as tg
from torch.utils.data import random_split
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from SE_torch.data_generator import DataGenerator
from SE_torch.net_preprocess.process_net_data import parse_ieee_mat, System, Branch


class ChannelScaler:
    def __init__(self, path=None, mean=None, std=None, eps=1e-10):
        if path is not None and os.path.exists(path):
            self.mean = torch.load(f'{path}/mean.pt')
            self.std = torch.load(f'{path}/std.pt')

        else:
            self.mean = mean
            self.std = std

        self.eps = eps

    def fit(self, x):
        self.mean = x.mean(dim=0)
        self.std = x.std(dim=0)
        self.std[self.std <= self.eps] = 1

    def transform(self, x):
        return (x - self.mean) / self.std

    def inverse(self, x):
        return x * self.std + self.mean

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.mean, f'{path}/mean.pt')
        torch.save(self.std, f'{path}/std.pt')


class GraphMeasurementsDataset(Dataset):
    def __init__(self, root=None, graphs=None):
        super().__init__()
        self.graphs = graphs
        if root is not None:
            files = sorted([p for p in Path(root).glob("sample_*.pt")])
            self.graphs = [torch.load(f, weights_only=False) for f in files]
        elif graphs is not None:
            self.graphs = graphs
        else:
            self.graphs = None

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]

    def save(self, root):
        save_dir = Path(root)
        save_dir.mkdir(parents=True, exist_ok=True)

        for i, graph in enumerate(self.graphs):
            torch.save(graph, save_dir / f"sample_{i}.pt")


def load_data(config, num_samples):
    file = config.get('file')
    if file is None:
        raise(ValueError('Please specify a net file'))

    data = parse_ieee_mat(file)
    system_data = data['data']['system']
    sys = System(system_data)
    branch = Branch(sys.branch)
    edge_index = torch.stack([branch.i, branch.j]).to(torch.long)
    if os.path.exists(config['data_path']):
        data = torch.load(f"{config['data_path']}/data.pt")
    else:
        scaler = ChannelScaler()
        data_generator = DataGenerator()
        T, V = data_generator.sample(sys, num_samples, random_flow=True, verbose=True)
        data = torch.concat([T, V], dim=1)
        scaler.fit(data)
        data = scaler.transform(data)

        os.makedirs(config['data_path'], exist_ok=True)
        torch.save(data, f"{config['data_path']}/data.pt")
        scaler.save(config['data_path'])

    graph_list = []

    for i in tqdm(range(num_samples), desc="Generating graphs", colour='MAGENTA', leave=False):
        v = data[i].reshape(2, -1).T
        graph_list.append(tg.data.Data(x=v, edge_index=edge_index))

    graph_dataset = GraphMeasurementsDataset(graphs=graph_list)

    n_train = int(0.8 * num_samples)
    n_test = num_samples - n_train
    generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(graph_dataset, [n_train, n_test], generator=generator)

    batch_size = config.get('batch_size', 1)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_dataset, test_dataset, sys.slk_bus


