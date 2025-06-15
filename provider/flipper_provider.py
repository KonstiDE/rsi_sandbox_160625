import os
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader

# Later, call this via dataset = FlipperDataset("train" or "valdiation")
class FlipperDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir

        self.data = np.load(os.path.join(data_dir, 'arrays.npz'))["arr_0"]
        self.labels = np.load(os.path.join(data_dir, 'labels.npz'))["arr_0"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_array = self.data[idx]
        label_array = self.labels[idx]

        return torch.Tensor(data_array), torch.Tensor(label_array)



def get_loader(data_dir):
    dataset = FlipperDataset(data_dir)

    return DataLoader(dataset, batch_size=1)
