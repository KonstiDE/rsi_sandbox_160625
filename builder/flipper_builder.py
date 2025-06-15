import os

import numpy as np
import config.config as config

from math import floor

def build_flipper_dataset():
    data = list(np.random.randint(0, 2, (1000, 10)))
    labels = [(d - 1) * -1 for d in data]

    training_how_many = floor(len(data) * config.get_training_amount())
    validation_how_many = floor(len(data) * config.get_validation_amount())

    # Put 70 percent of that in training
    files_0_70 = data[0:training_how_many]
    files_71_90 = data[training_how_many:training_how_many + validation_how_many]
    files_91_100 = data[training_how_many + validation_how_many:len(data)]

    # Copy for the labels
    labels_0_70 = labels[0:training_how_many]
    labels_71_90 = labels[training_how_many:training_how_many + validation_how_many]
    labels_91_100 = labels[training_how_many + validation_how_many:len(data)]

    print("Training dataset size: ", len(files_0_70))
    print("Validation dataset size: ", len(files_71_90))
    print("Test dataset size: ", len(files_91_100))

    np.savez(os.path.join(config.get_training_path(), "arrays.npz"), files_0_70)
    np.savez(os.path.join(config.get_training_path(), "labels.npz"), labels_0_70)

    np.savez(os.path.join(config.get_validation_path(), "arrays.npz"), files_71_90)
    np.savez(os.path.join(config.get_validation_path(), "labels.npz"), labels_71_90)

    np.savez(os.path.join(config.get_test_path(), "arrays.npz"), files_91_100)
    np.savez(os.path.join(config.get_test_path(), "labels.npz"), labels_91_100)

