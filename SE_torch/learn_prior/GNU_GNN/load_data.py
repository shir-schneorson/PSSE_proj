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


class GraphMeasurementsDataset(Dataset):
    def __init__(self, root=None, graphs=None):
        super().__init__()
        self.graphs = graphs
        if root is not None:
            files = sorted([p for p in Path(root).glob("*.pt")])
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
    if os.path.exists(config['data_path']):
        graph_dataset = GraphMeasurementsDataset(root=config['data_path'])
    else:

        branch = Branch(sys.branch)
        edge_index = torch.stack([branch.i, branch.j]).to(torch.long)
        data_generator = DataGenerator()
        data_gen_kwargs = config.get('data_gen_kwargs')

        graph_list = []
        for i in tqdm(range(num_samples), desc="Generating graphs", colour='MAGENTA'):
            gen_data = data_generator.generate_measurements(sys, branch, device=None, **data_gen_kwargs)
            z, _, meas_idx, _, _, _, T_true, V_true, _ = gen_data
            v = torch.stack([T_true, V_true], dim=1)
            graph_list.append(tg.data.Data(x=v, edge_index=edge_index, y=z))

        graph_dataset = GraphMeasurementsDataset(graphs=graph_list)
        graph_dataset.save(root=config['data_path'])
    n_train = int(0.8 * num_samples)
    n_test = num_samples - n_train
    train_dataset, test_dataset = random_split(graph_dataset, [n_train, n_test])

    batch_size = config.get('batch_size', 1)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_dataset, test_dataset, sys.slk_bus


