import os
import tempfile

from ray import tune
from ray import train as raytrain
from utils.stopper import PatienceStopper
from ray.train import Checkpoint

from metrics.mean_errors import mae

import torch
import torch.nn as nn

import config.config as cfg

import config.config as config
import statistics as s

from model.road_model import UNET
from provider.road_provider import get_loader

from loss.dice_sb_loss import DiceSBLoss

from tqdm import tqdm

device = "cuda:0"

def train(model, loader, optimizer, loss_fn, epoch, scaler):
    torch.enable_grad()
    model.train()

    losses = []
    accs = []

    loop = tqdm(loader)

    for (data, target) in loop:
        optimizer.zero_grad()

        data = model(data)
        data = torch.sigmoid(data)

        loss = loss_fn(data, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        accs.append(mae(data, target).item())

        losses.append(loss.item())

        loop.set_postfix_str("Epoch: {}".format(epoch))

    return s.mean(losses), s.mean(accs)


def validate(model, loader, loss_fn, epoch):
    model.eval()

    losses = []
    accs = []

    loop = tqdm(loader)

    for (data, target) in loop:
        data = model(data)
        data = torch.sigmoid(data)

        loss = loss_fn(data, target)

        losses.append(loss.item())

        loop.set_postfix_str("Epoch: {}".format(epoch))

        accs.append(mae(data, target).item())


    return s.mean(losses), s.mean(accs)


def run(ray_config):
    model = UNET()

    scaler = torch.amp.GradScaler()

    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=ray_config["lr"])

    train_loader = get_loader(config.get_training_path(), batch_size=ray_config["batch_size"])
    validation_loader = get_loader(config.get_validation_path(), batch_size=ray_config["batch_size"])

    for epoch in range(99):
        training_loss, training_acc = train(model, train_loader, optimizer, loss_function, epoch, scaler)
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
            "training_mae": training_acc,
            "validation_mae": validation_acc
        }

        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            torch.save(
                checkpoint_data,
                os.path.join(temp_checkpoint_dir, "model_epoch{}.pt".format(epoch)),
            )
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            raytrain.report(metrics, checkpoint=checkpoint)


if __name__ == '__main__':
    # run(ray_config={"lr": 1e-04, "batch_size": 1})

    stopper = PatienceStopper(
        metric="validation_loss",
        mode="min",
        patience=5
    )

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(run),
            resources={"cpu": 16, "gpu": 1},
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
