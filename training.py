from time import sleep

import torch
import torch.nn as nn

import config.config as config
import statistics as s

import matplotlib.pyplot as plt

from model.flipper_model import FlipperModel

from provider.flipper_provider import get_loader

from tqdm import tqdm


def train(model, loader, optimizer, loss_fn, epoch):
    torch.enable_grad()
    model.train()

    losses = []

    loop = tqdm(loader)

    for (data, target) in loop:
        optimizer.zero_grad()

        pred = model(data)

        loss = loss_fn(pred, target)
        losses.append(loss.item())

        loop.set_postfix_str("Epoch: {}".format(epoch))

        loss.backward()
        optimizer.step()

    return s.mean(losses)

def validate(model, loader, loss_fn, epoch):
    model.eval()

    losses = []

    loop = tqdm(loader)

    for (data, target) in loop:
        pred = model(data)

        loop.set_postfix_str("Epoch: {}".format(epoch))

        loss = loss_fn(pred, target)
        losses.append(loss.item())

    return s.mean(losses)


def run():
    model = FlipperModel(in_neurons=10, out_neurons=10)

    loss_function = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    train_loader = get_loader(config.get_training_path())
    validation_loader = get_loader(config.get_validation_path())

    training_losses = []
    validation_losses = []

    for epoch in range(200):
        training_loss = train(model, train_loader, optimizer, loss_function, epoch)
        validation_loss = validate(model, validation_loader, loss_function, epoch)

        training_losses.append(training_loss)
        validation_losses.append(validation_loss)

    plt.plot(training_losses)
    plt.plot(validation_losses)
    plt.show()

    array = torch.Tensor([0, 1, 0, 1, 1, 1, 1, 0, 0, 0])
    print(torch.round(model(array)))


if __name__ == '__main__':
    a = torch.randn((1, 3, 3))
    b = torch.randn((1, 3, 3))

    t = torch.cat((a, b), dim=0)
    print(t.shape)



