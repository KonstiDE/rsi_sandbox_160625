import numpy as np


def slice_n_dice(data, mask, t):
    rows, cols, channels = data.shape
    frames = []

    for r in range(0, rows, t):
        for c in range(0, cols, t):
            tile_data = data[r:r + t, c:c + t, :]
            tile_mask = mask[r:r + t, c:c + t, :]

            if (tile_data.shape[0], tile_data.shape[1]) != (t, t):
                padded_data = np.zeros((t, t, tile_data.shape[2]))
                padded_mask = np.zeros((t, t, 1))
                padded_data[:tile_data.shape[0], :tile_data.shape[1], :] = tile_data
                padded_mask[:tile_mask.shape[0], :tile_mask.shape[1], :] = tile_mask

                frames.append((padded_data, padded_mask))

            else:
                frames.append((tile_data, tile_mask))

    return frames


def raw_slicing_no_channels(data, t):
    rows, cols = data.shape
    frames = []

    for r in range(0, rows, t):
        for c in range(0, cols, t):
            tile_data = data[r:r + t, c:c + t]
            if tile_data.shape != (t, t):
                padded_data = np.zeros((t, t))
                padded_data[:tile_data.shape[0], :tile_data.shape[1]] = tile_data

                frames.append(padded_data)
            else:
                frames.append(tile_data)

    return frames
