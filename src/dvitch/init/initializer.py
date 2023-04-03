from abc import abstractmethod

import jax
import numpy as np

from ..types import TShapeLike


class Initializer:
    _seed: int
    _key: "jax.random.PRNGKey"

    @property
    def props(self) -> dict:
        return {}

    @property
    def seed(self) -> int:
        return self._seed

    def __repr__(self):
        (props := self.props).update(seed=self.seed)
        props = ", ".join(f"{k}={v}" for k, v in props.items())
        return f"<Initializer:{type(self).__name__}[{props}]>"

    def __init__(self, seed: int = None, *args, **kwargs):
        if args: raise TypeError(
            f"{type(self)}.__init__() takes only 2 positional argument ('self', 'seed') but got {len(args)} arg(s) extra"
        )
        if kwargs: raise TypeError(
            f"{type(self).__name__}.__init__() got unexpected keyword argument '{next(iter(kwargs))}'"
        )

        if seed is None:
            from dvitch.rand_seed import rand_seed
            seed = rand_seed()
        self._key = jax.random.PRNGKey(seed)
        self._seed = seed

    def __call__(self, *shape: "TShapeLike"):
        self._key, key = jax.random.split(self._key)
        return self._initialize(shape, key)

    @abstractmethod
    def _initialize(self, shape: "TShapeLike", key: "jax.random.PRNGKey"):
        raise NotImplementedError
