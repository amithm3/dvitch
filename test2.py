import time

import numpy as np
from jax import numpy as jnp

from dvitch.init import XavierNormal
from dvitch.loss import MSELoss
from dvitch.module import Module
from dvitch.parameter import Parameter, DParams
from dvitch.activ import ReLU, Sigmoid


class Linear(Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(XavierNormal()(out_features, in_features), 'weight')
        self.bias = Parameter(XavierNormal()(out_features), 'bias')

    def __forward__(self, params: DParams, x):
        weight = params.pop('weight')
        bias = params.pop('bias')
        return x @ weight.data.T + bias.data


class Sequential(Module):
    def __init__(self, *modules: Module):
        super().__init__()
        for idx, module in enumerate(modules):
            self.add_module(f"module_{idx}", module)

    def __forward__(self, params: DParams, x):
        for module_prefix, module in self.named_modules():
            x = module.__forward__(dict(self.decompose_params(params, module_prefix)), x)
        return x


if __name__ == '__main__':
    n = 10000
    ep = 10
    bs = 128
    i = jnp.array(np.random.rand(n, 784))
    t = jnp.array(np.random.rand(n, 10))
    s1 = Sequential(
        Linear(784, 256),
        ReLU(),
        Linear(256, 10),
        Sigmoid(),
    )
    lr = 0.1
    loss_func = MSELoss()

    print("Start training...")
    tm = time.time()

    for e in range(ep):
        for b in range(n // bs):
            mb = slice(b * bs, (b + 1) * bs)
            loss, g = s1.grad(loss_func, i[mb], t[mb])
            print(f"Epoch {e}, Batch {b}, Loss {loss:.4f}")
            for k, v in g.items():
                s1.get_parameter(k).data -= lr * v.data

    print(f"Time: {time.time() - tm:.4f}s")
