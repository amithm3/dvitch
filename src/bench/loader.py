import gzip

import numpy as np


def read_idxN_ubyte(N, file_path):
    with gzip.open(file_path, 'rb') as f:
        _ = int.from_bytes(f.read(4), 'big')  # magic number
        count = int.from_bytes(f.read(4), 'big')
        dims = [int.from_bytes(f.read(4), 'big') for _ in range(1, N)]

        data = f.read()
        array = np.frombuffer(data, dtype=np.uint8).reshape((count, *dims))
        return array


def read_idx4_ubyte(file_path):
    return read_idxN_ubyte(4, file_path)


def read_idx3_ubyte(file_path):
    return read_idxN_ubyte(3, file_path)


def read_idx1_ubyte(file_path):
    return read_idxN_ubyte(1, file_path)


def one_hot_encode(labels):
    assert len(labels.shape) == 1, \
        "Labels must be a 1D array"
    one_hot_encoded = np.zeros((labels.shape[0], _max := np.max(labels) + 1))
    one_hot_encoded[np.arange(labels.shape[0]), labels] = 1
    return one_hot_encoded, _max


def normalize(images, scale=1.0):
    return images / (_max := np.max(images)) * scale, _max
