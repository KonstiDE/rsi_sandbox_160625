import os
import tempfile

from ray import tune
from ray import train as raytrain
from utils.stopper import PatienceStopper
from ray.train import Checkpoint

import torch
import torch.nn as nn

import config.config as cfg

import config.config as config
import statistics as s

from model.road_model import UNET
from provider.road_provider import get_loader

from tqdm import tqdm


def train(model, loader, optimizer, loss_fn, epoch):
    torch.enable_grad()
    model.train()

    losses = []
    accs = []

    loop = tqdm(loader)

    for (data, target) in loop:
        optimizer.zero_grad()

        pred = model(data)

        loss = loss_fn(pred, target)
        losses.append(loss.item())

        class_predicted = torch.argmax(pred, dim=1)

        accs.append((class_predicted == target).float().mean().item())

        loop.set_postfix_str("Epoch: {}".format(epoch))

        loss.backward()
        optimizer.step()

    return s.mean(losses), s.mean(accs)


def validate(model, loader, loss_fn, epoch):
    model.eval()

    losses = []
    accs = []

    loop = tqdm(loader)

    for (data, target) in loop:
        pred = model(data)

        class_predicted = torch.argmax(pred, dim=1)

        loop.set_postfix_str("Epoch: {}".format(epoch))

        loss = loss_fn(pred, target)
        losses.append(loss.item())

        accs.append((class_predicted == target).float().mean().item())

    return s.mean(losses), s.mean(accs)


def run(ray_config):
    model = UNET()

    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=ray_config["lr"])

    train_loader = get_loader(config.get_training_path(), batch_size=ray_config["batch_size"])
    validation_loader = get_loader(config.get_validation_path(), batch_size=ray_config["batch_size"])

    for epoch in range(99):
        training_loss, training_acc = train(model, train_loader, optimizer, loss_function, epoch)
        validation_loss, validation_acc = validate(model, validation_loader, loss_function, epoch)

        # Ray report
        checkpoint_data = {
            "epoch": epoch,
            "net_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }

        metrics = {
            "training_loss": training_loss,
            "validation_loss": validation_loss,
            "training_acc": training_acc,
            "validation_acc": validation_acc
        }

        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            torch.save(
                checkpoint_data,
                os.path.join(temp_checkpoint_dir, "model_epoch{}.pt".format(epoch)),
            )
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            raytrain.report(metrics, checkpoint=checkpoint)


if __name__ == '__main__':
    run(ray_config={"lr": 1e-04, "batch_size": 1})

    # stopper = PatienceStopper(
    #     metric="validation_loss",
    #     mode="min",
    #     patience=5
    # )
    #
    # tuner = tune.Tuner(
    #     tune.with_resources(
    #         tune.with_parameters(run),
    #         resources={"cpu": 4, "gpu": 1},
    #     ),
    #     tune_config=tune.TuneConfig(
    #         metric="validation_loss",
    #         mode="min",
    #         num_samples=1,
    #     ),
    #     param_space=cfg.get_ray_config(),
    #     run_config=raytrain.RunConfig(
    #         stop=stopper,
    #         storage_path=os.path.join(cfg.get_ray_result_path(), "ray_results"),
    #     ),
    # )
    # results = tuner.fit()
    #
    # best_result = results.get_best_result("validation_loss", "min", "all")
    #
    # print("Best trial config: {}".format(best_result.config))
    # print("Best trial final validation loss: {}".format(best_result.metrics["validation_loss"]))
