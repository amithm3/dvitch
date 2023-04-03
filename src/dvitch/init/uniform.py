import jax

from .initializer import Initializer
from ..types import TShapeLike


class Uniform(Initializer):
    _minval: float
    _maxval: float

    @property
    def props(self) -> dict:
        return dict(minval=self._minval, maxval=self._maxval)

    def __init__(self, minval: float = 0, maxval: float = 1, *args, **kwargs):
        assert isinstance(minval, (float, int))
        assert isinstance(maxval, (float, int))
        super().__init__(*args, **kwargs)
        self._minval = minval
        self._maxval = maxval

    def _initialize(self, shape: "TShapeLike", key: "jax.random.PRNGKey"):
        return jax.random.uniform(key, shape, minval=self._minval, maxval=self._maxval)
