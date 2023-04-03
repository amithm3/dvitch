import jax
import numpy as np
from jax import lax
from jax.experimental import host_callback

from ..module import Module


class Dropout(Module):
    _p: float
    _seed: int
    _rng: jax.random.PRNGKey

    def __init__(self, p: float = 0.5, seed: int = None):
        assert 0 <= p <= 1
        if seed is None: seed = np.random.randint(0, 2 ** 32)
        super().__init__()
        self._p = float(p)
        self._p_bar = 1 / (1 - self._p) if self._p != 1 else None
        self._seed = seed
        self.register_buffer("_rng", jax.random.PRNGKey(seed))

    def change_key(self, key):
        self._rng = key

    def forward(self, _, buffs, x):
        rng = buffs.pop("_rng")
        if self._p_bar is None: return lax.zeros_like_array(x)
        if self.training:
            rng, subkey = jax.random.split(rng)
            host_callback.id_tap(lambda key, _: self.change_key(key), rng)
            mask = jax.random.bernoulli(subkey, self._p, x.shape)
            return lax.select(mask, lax.zeros_like_array(x), lax.mul(self._p_bar, x))
        return x
