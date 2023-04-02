import jax

from ..types import TShapeLike
from .initializer import Initializer


class Normal(Initializer):
    _mean: float
    _stddev: float

    @property
    def props(self) -> str:
        mean = self._mean
        stddev = self._stddev
        return f"{mean=}, {stddev=}"

    def __init__(self, mean: float = 0.0, stddev: float = 1.0, *args, **kwargs):
        assert isinstance(mean, (float, int))
        assert isinstance(stddev, (float, int)) and stddev > 0
        super().__init__(*args, **kwargs)
        self._mean = mean
        self._stddev = stddev

    def _initialize(self, shape: "TShapeLike", rng: "jax.random.PRNGKey"):
        return jax.random.normal(rng, shape) * self._stddev + self._mean
