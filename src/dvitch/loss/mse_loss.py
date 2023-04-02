from jax import lax

from .loss import Loss


class MSELoss(Loss):
    def _loss(self, outputs, *args, **kwargs):
        return lax.square(outputs - args[0]).mean()
