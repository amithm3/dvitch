from ..parameter import DParams
from ..module import Module


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        for idx, module in enumerate(modules):
            self.add_module(f"module_{idx}", module)

    def forward(self, params: "DParams", buffs, x):
        for module_prefix, module in self.named_modules():
            module_params = dict(self.decompose_params(params, module_prefix))
            module_buffs = dict(self.decompose_buffs(buffs, module_prefix))
            x = module.forward(module_params, module_buffs, x)
        return x


class Lambda(Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, _, __, x):
        return self.func(x)
