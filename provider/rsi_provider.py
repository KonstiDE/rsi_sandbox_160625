import os
import torch
import numpy as np
from PIL import Image

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(64)
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Later, call this via dataset = RSIDataset("train" or "valdiation")
class RSIDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir

        self.data = []
        self.labels = []

        for file in os.listdir(data_dir):
            self.data.append(Image.open(os.path.join(data_dir, file)))
            self.labels.append(get_class_from(file))

        # our data here is a list of (64, 64, 3 or 4) channeled numpy arrays
        # we want to filter out the 4th channel if that exists
        self.data = [np.asarray(i)[:, :, 0:3] for i in self.data]

        self.data = [transform(i) for i in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]

        return data, torch.tensor(label)


def get_loader(data_dir, batch_size=8):
    dataset = RSIDataset(data_dir)

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def get_class_from(file):
    split = file.split('_')

    if split[1] == "cloudy":
        return 0
    elif split[1] == "desert":
        return 1
    elif split[1] == "water":
        return 2
    else:
        # has to be green_area
        return 3
