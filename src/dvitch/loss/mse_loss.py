from jax import lax

from .loss import Loss


class MSELoss(Loss):
    def _loss(self, outputs, targets, *args, **kwargs):
        return lax.square(outputs - targets).mean()
