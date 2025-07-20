import os

import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms

from PIL import Image

class RoadDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir

        self.data = []
        self.labels = []

        for file in os.listdir(self.data_dir):
            array_file = np.load(os.path.join(data_dir, file))

            self.data.append(np.transpose(array_file["rgb"], (2, 0, 1)))
            self.labels.append(np.transpose(array_file["mask"], (2, 0, 1)))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_array = self.data[idx]
        label_array = self.labels[idx]

        return torch.Tensor(data_array), torch.Tensor(label_array)



def get_loader(data_dir, batch_size):
    dataset = RoadDataset(data_dir)

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
