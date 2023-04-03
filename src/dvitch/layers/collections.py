from ..parameter import DParams
from ..module import Module


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        for idx, module in enumerate(modules):
            self.add_module(f"module_{idx}", module)

    def forward(self, params: "DParams", buffs, inputs):
        for module_prefix, module in self.named_modules():
            module_params = dict(self.decompose_params(params, module_prefix))
            module_buffs = dict(self.decompose_buffs(buffs, module_prefix))
            inputs = module.forward(module_params, module_buffs, inputs)
        return inputs


class Lambda(Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward_nm(self, x):
        return self.func(x)
