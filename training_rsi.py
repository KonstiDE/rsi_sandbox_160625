import shutup
shutup.please()

import torch
import torch.nn as nn

import numpy as np

import config.config as config
import statistics as s

import matplotlib.pyplot as plt

from model.rsi_model import RSIModel

from provider.rsi_provider import get_loader

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

        loss = loss_fn(pred, target)
        losses.append(loss.item())

        loop.set_postfix_str("Epoch: {}".format(epoch))

    return s.mean(losses)


def run():
    model = RSIModel(out_classes=4)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

    train_loader = get_loader(config.get_training_path())
    validation_loader = get_loader(config.get_validation_path())

    training_losses = []
    validation_losses = []

    for epoch in range(10):
        training_loss = train(model, train_loader, optimizer, loss_function, epoch)
        validation_loss = validate(model, validation_loader, loss_function, epoch)

        training_losses.append(training_loss)
        validation_losses.append(validation_loss)

        print("\n---")
        print("Losses were:")
        print("Training: {}".format(training_loss))
        print("Validation: {}".format(validation_loss))
        print("---\n")

    plt.plot(training_losses)
    plt.plot(validation_losses)
    plt.show()

    torch.save(model.state_dict(), "models/rsi_model_20.pt")


def test():
    saved_state_model = torch.load("models/rsi_model_20.pt")
    model = RSIModel(out_classes=4)
    model.load_state_dict(saved_state_model)
    model.eval()

    test_loader = get_loader(config.get_test_path(), batch_size=1)

    correctly_predicted = 0

    loop = tqdm(test_loader)

    for (data, target) in loop:
        pred = model(data)

        class_predicted = torch.argmax(pred, dim=1)

        if class_predicted == target.item():
            correctly_predicted += 1

    print("Accuracy on test set: {}".format(correctly_predicted / len(test_loader.dataset)))


if __name__ == '__main__':
    run()
