from jax import lax

from ..module import Module
from ..init import Uniform


class ReLU(Module):
    def __forward__(self, _, x):
        return lax.select(x > 0, x, lax.zeros_like_array(x))


class LeakyReLU(Module):
    _negative_slope: float

    @property
    def props(self) -> str:
        negative_slope = self._negative_slope
        return f"{negative_slope=}"

    def __init__(self, negative_slope: float = 0.01):
        assert isinstance(negative_slope, (float, int)) and negative_slope > 0
        super().__init__()
        self._negative_slope = float(negative_slope)

    def __forward__(self, _, x):
        return x * (x > 0) + self._negative_slope * x * (x < 0)


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

    def __forward__(self, _, x):
        return x * (x > 0) + self._negative_slope * x * (x < 0)


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

    def __forward__(self, _, x):
        return x * (x > 0) + self._rand(x.shape) * x * (x < 0)


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

    def __forward__(self, _, x):
        return x * (x > 0) + self._alpha * (lax.exp(x) - 1) * (x < 0)


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

    def __forward__(self, _, x):
        return self._scale * x * (x > 0) + self._scale * self._alpha * (lax.exp(x) - 1) * (x < 0)
