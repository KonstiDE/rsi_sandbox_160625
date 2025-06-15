import torch.nn as nn


class FlipperModel(nn.Module):
    def __init__(self, in_neurons=10, out_neurons=10):
        super(FlipperModel, self).__init__()

        self.in_neurons = in_neurons
        self.out_neurons = out_neurons

        self.sequential = nn.Sequential(
            nn.Linear(in_neurons, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, out_neurons),
            nn.ReLU()
        )


    def forward(self, x):
        return self.sequential(x)