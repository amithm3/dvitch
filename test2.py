from jax import numpy as jnp

import dvitch as nn


if __name__ == "__main__":
    m = nn.Dropout(0)
    m.train(True)
    print(m(jnp.ones((3, 3))))
