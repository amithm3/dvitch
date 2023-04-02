from typing import Union, Any

from jax import Array
from numpy import ndarray

TTensor = Union[Array, ndarray]
TTensorLike = Union[TTensor, float]

TShape = tuple[int, ...]
TShapeLike = Union[TShape, Any]
