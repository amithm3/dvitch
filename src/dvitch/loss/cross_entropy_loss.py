from jax import lax
import jax.numpy as jnp

from .loss import Loss


# todo: implement cross entropy loss
class CrossEntropyLoss(Loss):
    def _loss(self, outputs, targets, *args, **kwargs):
        return lax.neg(jnp.sum(lax.mul(targets, lax.log(outputs))))
