from jax import lax, ShapeDtypeStruct
from jax.experimental import host_callback

from ..module import Module
from ..init import Uniform


class ReLU(Module):
    def forward(self, _, __, x):
        return lax.select(x > 0, x, lax.zeros_like_array(x))


class LeakyReLU(Module):
    @property
    def props(self) -> str:
        negative_slope = 1e-2
        return f"{negative_slope=}"

    def __init__(self):
        super().__init__()

    def forward(self, _, __, x):
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

    def forward(self, _, __, x):
        return lax.select(x > 0, x, lax.mul(self._negative_slope, x))


class RReLU(Module):
    _lower: float
    _upper: float

    @property
    def props(self) -> str:
        lower = self._lower
        upper = self._upper
        return f"{lower=}, {upper=}"

    def __init__(self, lower: float = 0.125, upper: float = 0.3333333333333333):
        assert isinstance(lower, (float, int)) and lower > 0
        assert isinstance(upper, (float, int)) and upper > 0
        super().__init__()
        self._lower = float(lower)
        self._upper = float(upper)
        self._rand = Uniform(self._lower, self._upper)

    def forward(self, _, __, x):
        rnd = host_callback.call(lambda x_: self._rand(*x.shape), x, result_shape=ShapeDtypeStruct(x.shape, x.dtype))
        return lax.select(x > 0, x, lax.mul(rnd, x))


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

    def forward(self, _, __, x):
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

    def forward(self, _, __, x):
        return lax.select(x > 0,
                          lax.mul(self._scale, x),
                          lax.mul(lax.mul(self._scale, self._alpha), lax.sub(lax.exp(x), 1)))
