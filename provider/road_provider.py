import os
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms

from PIL import Image

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(512)
])

class RoadDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir

        self.data = []
        self.labels = []

        for file in os.listdir(self.data_dir):
            self.data.append(Image.open(os.path.join(self.data_dir, file)))
            self.labels.append(Image.open(os.path.join(self.data_dir + "_labels", file[:-1])))


        self.data = [np.asarray(i) for i in self.data]
        self.labels = [np.asarray(i) for i in self.labels]
        self.data = [transform(e) for e in self.data]
        self.labels = [transform(e) for e in self.labels]


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_array = self.data[idx]
        label_array = self.labels[idx]

        return data_array, torch.tensor(label_array)



def get_loader(data_dir, batch_size):
    dataset = RoadDataset(data_dir)

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
