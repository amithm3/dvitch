from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from functools import partial
from typing import Callable, Iterator, TypeVar

from jax import jit, value_and_grad

from dvitch.parameter import Parameter, ParameterLike, INParameter, IParameter, DParams
from dvitch.types import TTensorLike

T = TypeVar("T")


class Module(metaclass=ABCMeta):
    _parameters: OrderedDict[str, "Parameter"]
    _modules: OrderedDict[str, "Module"]
    _buffers: OrderedDict[str, "TTensorLike"]

    @property
    def classname(self) -> str:
        return type(self).__name__

    @property
    def name(self) -> str:
        return ""

    @property
    def props(self) -> str:
        return ""

    def __repr__(self):
        return f"<{self.classname}{f':{self.name}' if self.name else ''}[{self.props}]>"

    def __str__(self):
        return self.__repr__()

    def __init__(self, *args, **kwargs):
        if args: raise TypeError(
            f"{type(self)}.__init__() takes only 1 positional argument ('self',) but got {len(args)} arg(s) extra"
        )
        if kwargs: raise TypeError(
            f"{type(self).__name__}.__init__() got unexpected keyword argument '{next(iter(kwargs))}'"
        )

        super().__setattr__("_parameters", OrderedDict())
        super().__setattr__("_modules", OrderedDict())
        super().__setattr__("_buffers", OrderedDict())

    def __setattr__(self, key, value):
        if isinstance(value, Parameter):
            if key in self._parameters:
                self._parameters[key].update(value)
                return
            self.register_parameter(key, value)
        elif isinstance(value, Module):
            self.add_module(key, value)
        else:
            if key in self._buffers:
                if value is None or isinstance(value, TTensorLike):
                    self._buffers[key] = value
                    return
                raise TypeError(f"Cannot assign {type(value)} to {type(self)} buffer")
            return super().__setattr__(key, value)

    def __getattr__(self, key):
        if hasattr(self, "_parameters") and key in self._parameters:
            return self._parameters[key]
        elif hasattr(self, "_modules") and key in self._modules:
            return self._modules[key]
        elif hasattr(self, "_buffers") and key in self._buffers:
            return self._buffers[key]
        else:
            raise AttributeError(f"{type(self).__name__} object has no attribute '{key}'")

    def get_module(self, ladder: str):
        bars = ladder.split(".")
        atom = self

        for bar in bars:
            if not hasattr(atom, bar):
                raise AttributeError(f"{type(atom).__name__} object has no attribute '{bar}'")
            atom = getattr(atom, bar)
            if not isinstance(atom, Module):
                raise TypeError(f"'{bar}' is not a Module")

        return atom

    def get_parameter(self, ladder: str):
        ladder, param_name = ladder.rsplit(".", 1)
        module = self.get_module(ladder)

        if not hasattr(module, param_name):
            raise AttributeError(f"{type(module).__name__} object has no attribute '{param_name}'")
        param = getattr(module, param_name)
        if not isinstance(param, Parameter):
            raise TypeError(f"'{param_name}' is not a Parameter")

        return param

    @staticmethod
    def decompose_params(params: DParams, prefix: str):
        for key, param in params.items():
            if key.startswith(prefix):
                yield key.removeprefix(prefix + '.'), param

    def add_module(self, name: str, module: "Module"):
        assert name and isinstance(name, str), \
            f"Argument 'name' must be a non empty 'str'"
        assert module is None or isinstance(module, Module), \
            f"Argument 'module' must be of type '{Module}' or 'None', not {type(module)}"
        assert name not in self._modules, \
            f"Module named '{name}' already exists"
        if not hasattr(self, "_modules"): raise AttributeError(
            f"Cannot assign Module to {type(self)} before it has been initialized"
        )
        self._modules[name] = module

    def pop_module(self, name: str):
        assert name in self._modules, \
            f"Module named '{name}' does not exist"
        return self._modules.pop(name)

    def register_parameter(self, name: str, parameter: "ParameterLike"):
        assert name and isinstance(name, str), \
            f"Argument 'name' must be a non empty 'str'"
        assert isinstance(parameter, ParameterLike), \
            f"Argument 'parameter' must be of type '{ParameterLike}', not {type(parameter)}"
        assert name not in self._parameters, \
            f"Parameter named '{name}' already exists"
        if not isinstance(parameter, Parameter): parameter = Parameter(parameter)
        if not hasattr(self, "_parameters"): raise AttributeError(
            f"Cannot assign Parameter to {type(self)} before it has been initialized"
        )
        self._parameters[name] = parameter

    def deregister_parameter(self, name: str):
        assert name in self._parameters, \
            f"Parameter '{name}' does not exist"
        return self._parameters.pop(name)

    def register_buffer(self, name: str, buffer: "TTensorLike"):
        assert name and isinstance(name, str), \
            f"Argument 'name' must be a non empty 'str'"
        assert buffer is None or isinstance(buffer, TTensorLike), \
            f"Argument 'buffer' must be of type '{TTensorLike}' or 'None', not {type(buffer)}"
        assert name not in self._buffers, \
            f"Buffer named '{name}' already exists"
        if not hasattr(self, "_buffers"): raise AttributeError(
            f"Cannot assign Buffer to {type(self)} before it has been initialized"
        )
        self._buffers[name] = buffer

    def deregister_buffer(self, name: str):
        assert name in self._buffers, \
            f"Buffer '{name}' does not exist"
        return self._buffers.pop(name)

    def named_modules(
            self,
            prefix: str = '',
            include_self: bool = False,
            include_children: bool = True,
    ) -> "INModule":
        if include_self: yield prefix, self
        for key, module in self._modules.items():
            module_prefix = f"{prefix}.{key}" if prefix else key
            if include_children:
                yield from module.named_modules(module_prefix, include_self=True)
            else:
                yield module_prefix, module

    def modules(self, include_children: bool = True) -> "IModule":
        for _, module in self.named_modules(include_children=include_children):
            yield module

    def _named_members(
            self,
            get_members_fn: Callable[["Module"], dict[str, T]],
            prefix: str = '',
            recursive: bool = True,
            remove_duplicates: bool = False,
    ) -> Iterator[tuple[str, T]]:
        memo = set()
        for module_prefix, module in self.named_modules(prefix, include_self=True, include_children=recursive):
            members = get_members_fn(module)
            for key, member in members:
                if remove_duplicates and member in memo: continue
                memo.add(member)
                submodule_prefix = f"{module_prefix}.{key}" if module_prefix else key
                yield submodule_prefix, member

    def named_parameters(
            self,
            prefix: str = '',
            recursive: bool = True,
            remove_duplicates: bool = False
    ) -> INParameter:
        yield from self._named_members(
            lambda module: getattr(module, "_parameters").items(),
            prefix=prefix, recursive=recursive, remove_duplicates=remove_duplicates,
        )

    def parameters(self, recursive: bool = True) -> IParameter:
        for _, parameter in self.named_parameters(recursive=recursive):
            yield parameter

    def named_buffers(
            self,
            prefix: str = '',
            recursive: bool = True,
            remove_duplicates: bool = False
    ) -> Iterator[tuple[str, TTensorLike]]:
        yield from self._named_members(
            lambda module: getattr(module, "_buffers").items(),
            prefix=prefix, recursive=recursive, remove_duplicates=remove_duplicates,
        )

    def buffers(self, recursive: bool = True) -> Iterator[TTensorLike]:
        for _, buffer in self.named_buffers(recursive=recursive):
            yield buffer

    @abstractmethod
    def __forward__(self, params: DParams, inputs):
        raise NotImplementedError

    def __call__(self, inputs):
        return self.__forward__(dict(self.named_parameters()), inputs)

    @partial(jit, static_argnums=(0, 2))
    def __grad__(self, params: DParams, scalar_function, inputs, *args, **kwargs):
        @value_and_grad
        def _func(_params: DParams, *_args, **_kwargs):
            return scalar_function(self.__forward__(_params, inputs), *_args, **_kwargs)

        return _func(params, *args, **kwargs)

    def grad(self, scalar_function, inputs, *args, **kwargs):
        return self.__grad__(dict(self.named_parameters()), scalar_function, inputs, *args, **kwargs)


IModule = Iterator[Module]
INModule = Iterator[tuple[str, Module]]
