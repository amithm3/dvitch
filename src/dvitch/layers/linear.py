from .. import init
from ..init import Initializer
from ..module import Module
from ..parameter import Parameter, DParams


class Linear(Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            *,
            use_bias: bool = True,
            initializer: "Initializer" = None,
    ):
        if initializer is None: initializer = init.XavierNormal()
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.initializer = initializer

        self.weight = Parameter(None, "weight")
        if use_bias: self.bias = Parameter(None, name="bias")
        else: self.bias = None
        self.__init_params__()

    def __init_params__(self):
        self.weight.data = self.initializer(self.out_features, self.in_features)
        if self.bias is not None: self.bias.data = self.initializer(self.out_features)

    def forward_nb(self, params: "DParams", inputs):
        weight = params["weight"]
        if self.bias is not None:
            bias = params["bias"]
            return inputs @ weight.data.T + bias.data
        return inputs @ weight.data.T
