from typing import TypeVar

import jax

from .normal import Normal
from .uniform import Uniform
from .initializer import Initializer
from ..types import TShapeLike


T = TypeVar("T", bound=Initializer)


def make_xavier_class(
        name: str,
        base: type[T],
):
    assert isinstance(name, str)
    assert issubclass(base, Initializer)

    class Xavier(base):
        @property
        def props(self) -> dict:
            return dict(he=self._he, **super().props)

        def __init__(self, he: float = 1.0, *args, **kwargs):
            assert isinstance(he, (float, int)) and he > 0
            super().__init__(*args, **kwargs)
            self._he = he

        def _initialize(self, shape: "TShapeLike", key: "jax.random.PRNGKey"):
            return super()._initialize(shape, key) * (self._he / (shape[0] + shape[-1])) ** 0.5

    Xavier.__name__ = name

    return Xavier


XavierNormal = make_xavier_class("XavierNormal", Normal)
XavierUniform = make_xavier_class("XavierUniform", Uniform)
