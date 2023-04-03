import numpy as np


def rand_seed():
    return int(np.random.randint(0, 1 << 32 - 1))
