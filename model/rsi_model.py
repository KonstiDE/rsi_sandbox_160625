import torch
import torch.nn as nn


class RSIModel(nn.Module):
    def __init__(self, out_classes=4):
        super(RSIModel, self).__init__()

        self.resnet18 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)

        num_features = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_features, out_classes)

    def forward(self, x):
        return self.resnet18(x)
