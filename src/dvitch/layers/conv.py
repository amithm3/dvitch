from typing import Literal

from jax import numpy as jnp

from .. import DParams, Parameter
from ..module import Module


class ConvNd(Module):
    padding_modes = {"zeros", "reflect", "replicate", "circular"}

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: tuple[int, ...],
            stride: tuple[int, ...],
            padding: tuple[int, ...],
            dilation: tuple[int, ...],
            output_padding: tuple[int, ...],
            groups: int,
            use_bias: bool,
            padding_mode: Literal["zeros", "reflect", "replicate", "circular"],
    ):
        assert padding_mode in self.padding_modes
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.output_padding = output_padding
        self.groups = groups

        if use_bias:
            self.bias = Parameter(jnp.zeros((out_channels,)))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward_nb(self, params: "DParams", inputs):
        pass
