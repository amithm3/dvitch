import jax.random
from jax import numpy as jnp, jit, ShapeDtypeStruct
from jax.experimental import host_callback


def keys(seed):
    key = jax.random.PRNGKey(seed)
    while True:
        key, subkey = jax.random.split(key)
        yield subkey


key_gen = keys(0)


def rng(shape):
    return jax.random.uniform(next(key_gen), shape)


@jit
def foo(x):
    rnd = host_callback.call(rng, x.shape, result_shape=ShapeDtypeStruct(x.shape, jnp.float32))
    return x * rnd


if __name__ == "__main__":
    print(foo(jnp.ones((3, 3))))
