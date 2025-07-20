import os

import numpy as np

import config.config as cfg

from utils.tools import slice_n_dice

from PIL import Image
from tqdm import tqdm

import uuid

if __name__ == '__main__':
    for data_part in ["train", "val", "test"]:
        data_path = os.path.join(cfg.get_base_path(), "data/roads", data_part)
        label_path = os.path.join(cfg.get_base_path(), "data/roads", data_part + "_labels")

        data_images = os.listdir(str(data_path))
        label_images = os.listdir(str(label_path))

        frames = []

        for (d_file, l_file) in zip(sorted(data_images), sorted(label_images)):
            d_file = np.asarray(Image.open(os.path.join(str(data_path), d_file)))
            l_file = np.asarray(Image.open(os.path.join(str(label_path), l_file)))

            l_file = np.expand_dims(l_file, 2)

            tiles = slice_n_dice(d_file, l_file, 512)

            frames.extend(tiles)

        for frame in tqdm(frames):
            np.savez(os.path.join(cfg.get_base_path(), "frames", data_part, str(uuid.uuid4()) + ".npz"), rgb=frame[0], mask=frame[1])
