from jax import lax, numpy as jnp

from ..module import Module


class SoftMax(Module):
    _axis: int

    @property
    def props(self):
        axis = self._axis
        return f"{axis=}"

    def __init__(self, axis=-1):
        super().__init__()
        self._axis = axis

    def forward_nm(self, x):
        x_max = jnp.max(x, self._axis, keepdims=True, initial=0)
        un_norm = jnp.exp(x - lax.stop_gradient(x_max))
        return un_norm / jnp.sum(un_norm, self._axis, keepdims=True, initial=0)


class SoftPlus(Module):
    def forward_nm(self, x):
        x_max = jnp.max(x, keepdims=True, initial=0)
        return lax.log(lax.add(1., lax.exp(x - lax.stop_gradient(x_max))))


class SoftSign(Module):
    def forward_nm(self, x):
        return lax.div(x, lax.add(1., lax.abs(x)))
