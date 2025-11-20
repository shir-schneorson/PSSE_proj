import os

import torch
from torch.utils.data import TensorDataset, Subset, DataLoader
from sklearn.model_selection import train_test_split

from SE_torch.data_generator import DataGenerator

BATCH_SIZE = 512
DATA_DIM = 235
HALF_DATA_DIM = 118

def load_data(sys, n_samples, batch_size=BATCH_SIZE):
    if os.path.exists('../datasets/data.pt'):
        data = torch.load('../datasets/data.pt')
    else:
        data_generator = DataGenerator()
        T, V = data_generator.sample(sys, num_samples=n_samples, random_flow=True)
        data = torch.concat([T, V] , dim=1)
        data = torch.tensor(data)
        torch.save(data, '../datasets/data.pt')

    data = torch.concat([data[:, :sys.slk_bus[0]], data[:, sys.slk_bus[0] + 1:]], dim=1)
    mean, std, cov = data.mean(0), data.std(0), data.T.cov()
    std[std < 1e-10] = 1
    cov += torch.eye(DATA_DIM)
    torch.save(mean, '../datasets/mean.pt')
    torch.save(std, '../datasets/std.pt')
    torch.save(cov, '../datasets/cov.pt')
    data = (data - mean) / std
    dataset = TensorDataset(data)


    train_idx, test_idx = train_test_split(range(n_samples), train_size=0.75, test_size=0.25)

    train_dataset = Subset(dataset, train_idx)
    test_dataset = Subset(dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_dataset, test_dataset, mean, cov, std