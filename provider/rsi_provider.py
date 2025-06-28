import os
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms

from PIL import Image

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(64)
])


class RSIDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir

        self.data = []
        self.labels = []

        for file in os.listdir(self.data_dir):
            self.data.append(Image.open(os.path.join(self.data_dir, file)))
            self.labels.append(get_class_from_name(file))

        self.data = [np.asarray(e)[:, :, 0:3] for e in self.data]
        self.data = [transform(e) for e in self.data]


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_array = self.data[idx]
        label_array = self.labels[idx]

        #      (256, 256, 3) (3)
        return data_array,   torch.tensor(label_array)



def get_loader(data_dir, batch_size):
    dataset = RSIDataset(data_dir)

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def get_class_from_name(filename):
    split = filename.split('_')

    if split[1] == "cloudy":
        return 0
    elif split[1] == "desert":
        return 1
    elif split[1] == "water":
        return 2
    else:
        return 3
