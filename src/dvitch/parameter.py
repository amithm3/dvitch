from abc import ABCMeta
from typing import Union, Iterator

import jax
import jax.numpy as jnp

from .types import TTensor, TTensorLike


class _ParameterMeta(ABCMeta):
    def __instancecheck__(self, instance):
        return super().__instancecheck__(instance)


class Parameter(metaclass=_ParameterMeta):
    _data: "TTensor"
    _name: str

    @property
    def name(self):
        return self._name

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value: TTensor):
        assert isinstance(value, TTensor)
        assert value.shape == self._data.shape
        assert value.dtype == self._data.dtype
        self._data = value

    def __repr__(self):
        shape = self._data.shape
        return f"<Parameter{f':{self.name}' if self.name else ''}{[*shape]}>"

    def __init__(self, data: "TTensor", name: str = None, *, safe_copy: bool = True):
        if safe_copy:
            self._data = jnp.copy(data).astype(jnp.float32)
        else:
            self._data = data
        self._name = name

    def _tree_flatten(self):
        children = (self._data,)
        aux_data = (self._name,)
        return children, aux_data

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(data=children[0], name=aux_data[0], safe_copy=False)


jax.tree_util.register_pytree_node(Parameter, Parameter._tree_flatten, Parameter._tree_unflatten)

ParameterLike = Union[Parameter, TTensorLike, None]
INParameter = Iterator[tuple[str, Parameter]]
DParams = dict[str, Parameter]
IParameter = Iterator[Parameter]
