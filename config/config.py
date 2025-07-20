import os

import torch.nn
from ray import tune


config = {
    "base_path": "/home/caipi/PycharmProjects/rsi_sandbox_160625/",
    "training_path": "data/roads/train",
    "validation_path": "data/roads/val",
    "test_path": "data/rsi/test",
    "training_amount_percentage": 0.7,
    "validation_amount_percentage": 0.2,
    "test_amount_percentage": 0.1,
    "ray_result_path": "/home/caipi/PycharmProjects/rsi_sandbox_160625/",
    "ray_config": {
        "lr": tune.grid_search([1e-04, 1e-05]),
        "batch_size": tune.grid_search([4, 8, 16])
    },
}

def get_training_path():
    return os.path.join(config["base_path"], config["training_path"])

def get_validation_path():
    return os.path.join(config["base_path"], config["validation_path"])

def get_test_path():
    return os.path.join(config["base_path"], config["test_path"])

def get_training_amount():
    return config["training_amount_percentage"]

def get_validation_amount():
    return config["validation_amount_percentage"]

def get_test_amount():
    return config["test_amount_percentage"]

def get_ray_result_path():
    return config["ray_result_path"]

def get_ray_config():
    return config["ray_config"]

def get_base_path():
    return config["base_path"]