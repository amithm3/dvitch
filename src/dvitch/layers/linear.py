from jax import numpy as jnp

from .. import init
from ..init import Initializer
from ..module import Module
from ..parameter import Parameter, DParams


class Linear(Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            use_bias: bool = True,
            initializer: "Initializer" = None,
    ):
        if initializer is None: initializer = init.XavierNormal()
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.initializer = initializer

        self.weight = Parameter(jnp.zeros((out_features, in_features)), name="weight")
        if use_bias:
            self.bias = Parameter(jnp.zeros(out_features), name="bias")
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data = self.initializer(self.out_features, self.in_features)
        if self.bias is not None: self.bias.data = self.initializer(self.out_features)

    def forward_nb(self, params: "DParams", inputs):
        weight = params.pop("weight")
        bias = params.pop("bias")
        if bias is not None:
            return inputs @ weight.data.T + bias.data
        return inputs @ weight.data.T
