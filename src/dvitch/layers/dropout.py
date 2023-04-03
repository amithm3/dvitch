import jax
from jax import lax
from jax.experimental import host_callback

from ..module import Module


class Dropout(Module):
    _p: float
    _seed: int
    _key: jax.random.PRNGKey

    def __init__(self, p: float = 0.5, seed: int = None):
        assert 0 <= p <= 1
        if seed is None:
            from dvitch.rand_seed import rand_seed
            seed = rand_seed()
        super().__init__()
        self._p = float(p)
        self._p_bar = 1 / (1 - self._p) if self._p != 1 else None
        self._seed = seed
        self.register_buffer("_key", jax.random.PRNGKey(seed))

    def change_key(self, key):
        self._key = key

    def forward_np(self, buffs, x):
        key = buffs.pop("_key")
        if self._p_bar is None: return lax.zeros_like_array(x)
        if self.training:
            key, subkey = jax.random.split(key)
            host_callback.id_tap(lambda k, _: self.change_key(k), key)
            mask = jax.random.bernoulli(subkey, self._p, x.shape)
            return lax.select(mask, lax.zeros_like_array(x), lax.mul(self._p_bar, x))
        return x
