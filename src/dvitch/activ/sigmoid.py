from jax import lax

from ..module import Module


class Sigmoid(Module):
    _smooth: float
    _offset: float

    @property
    def props(self) -> str:
        smooth = self._smooth
        offset = self._offset
        return f"{smooth=}, {offset=}"

    def __init__(self, smooth: float = 1.0, offset: float = 0.0):
        assert isinstance(smooth, (float, int)) and smooth > 0
        assert isinstance(offset, (float, int)) and offset >= 0
        super().__init__()
        self._smooth = float(smooth)
        self._offset = float(offset)

    def __forward__(self, _, x):
        exp = lax.exp(lax.add(lax.mul(lax.neg(self._smooth), x), self._offset))
        return lax.div(1., lax.add(1., exp))
