import jax
from jax import lax
from jax.experimental import host_callback

from ..module import Module


class ReLU(Module):
    def forward_nm(self, x):
        return lax.select(x > 0, x, lax.zeros_like_array(x))


class LeakyReLU(Module):
    @property
    def props(self) -> str:
        negative_slope = 1e-2
        return f"{negative_slope=}"

    def __init__(self):
        super().__init__()

    def forward_nm(self, x):
        return lax.select(x > 0, x, lax.mul(1e-2, x))


class PReLu(Module):
    _negative_slope: float

    @property
    def props(self) -> str:
        negative_slope = self._negative_slope
        return f"{negative_slope=}"

    def __init__(self, negative_slope: float = 0.01):
        assert isinstance(negative_slope, (float, int)) and negative_slope > 0
        super().__init__()
        self._negative_slope = float(negative_slope)

    def forward_nm(self, x):
        return lax.select(x > 0, x, lax.mul(self._negative_slope, x))


class RReLU(Module):
    _lower: float
    _upper: float
    _seed: int
    _key: jax.random.PRNGKey

    @property
    def props(self) -> str:
        lower = self._lower
        upper = self._upper
        return f"{lower=}, {upper=}"

    def __init__(self, lower: float = 1 / 8, upper: float = 1 / 3, seed: int = None):
        assert isinstance(lower, (float, int)) and lower > 0
        assert isinstance(upper, (float, int)) and upper > 0
        assert lower < upper
        if seed is None:
            from dvitch.rand_seed import rand_seed
            seed = rand_seed()
        super().__init__()
        self._lower = float(lower)
        self._upper = float(upper)
        self._seed = int(seed)
        self.register_buffer("_key", jax.random.PRNGKey(seed))

    def change_key(self, key):
        self._key = key

    def forward_np(self, buffs, x):
        key = buffs.pop("_key")
        key, subkey = jax.random.split(key)
        host_callback.id_tap(lambda k, _: self.change_key(k), key)
        rand = jax.random.uniform(subkey, x.shape, minval=self._lower, maxval=self._upper)
        return lax.select(x > 0, x, lax.mul(rand, x))


class ELU(Module):
    _alpha: float

    @property
    def props(self) -> str:
        alpha = self._alpha
        return f"{alpha=}"

    def __init__(self, alpha: float = 1.0):
        assert isinstance(alpha, (float, int)) and alpha > 0
        super().__init__()
        self._alpha = float(alpha)

    def forward_nm(self, x):
        return lax.select(x > 0, x, lax.mul(self._alpha, lax.sub(lax.exp(x), 1)))


class SELU(Module):
    _alpha: float
    _scale: float

    @property
    def props(self) -> str:
        alpha = self._alpha
        scale = self._scale
        return f"{alpha=}, {scale=}"

    def __init__(
            self,
            alpha: float = 1.6732632423543772848170429916717,
            scale: float = 1.0507009873554804934193349852946
    ):
        assert isinstance(alpha, (float, int)) and alpha > 0
        assert isinstance(scale, (float, int)) and scale > 0
        super().__init__()
        self._alpha = float(alpha)
        self._scale = float(scale)

    def forward_nm(self, x):
        return lax.select(x > 0,
                          lax.mul(self._scale, x),
                          lax.mul(lax.mul(self._scale, self._alpha), lax.sub(lax.exp(x), 1)))
