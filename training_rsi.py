import os
from ray import tune
from ray import train as raytrain
from utils.stopper import PatienceStopper

import torch
import torch.nn as nn

import config.config as cfg

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

        loop.set_postfix_str("Epoch: {}".format(epoch))

        loss = loss_fn(pred, target)
        losses.append(loss.item())

    return s.mean(losses)


def run(ray_config):
    model = RSIModel(out_classes=4)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=ray_config["lr"])

    train_loader = get_loader(config.get_training_path(), batch_size=ray_config["batch_size"])
    validation_loader = get_loader(config.get_validation_path(), batch_size=ray_config["batch_size"])

    training_losses = []
    validation_losses = []

    for epoch in range(10):
        training_loss = train(model, train_loader, optimizer, loss_function, epoch)
        validation_loss = validate(model, validation_loader, loss_function, epoch)

        training_losses.append(training_loss)
        validation_losses.append(validation_loss)

    plt.plot(training_losses)
    plt.plot(validation_losses)
    plt.show()

    torch.save(model.state_dict(), "models/rsi_model_10.pt")



if __name__ == '__main__':
    stopper = PatienceStopper(
        metric="validation_loss",
        mode="min",
        patience=5
    )

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(run),
            resources={"cpu": 4},
        ),
        tune_config=tune.TuneConfig(
            metric="validation_loss",
            mode="min",
            num_samples=1,
        ),
        param_space=cfg.get_ray_config(),
        run_config=raytrain.RunConfig(
            stop=stopper,
            storage_path=os.path.join(cfg.get_ray_result_path(), "ray_results"),
        ),
    )
    results = tuner.fit()

    best_result = results.get_best_result("validation_loss", "min", "all")

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(best_result.metrics["validation_loss"]))



