from abc import ABCMeta, abstractmethod


class Loss(metaclass=ABCMeta):
    @property
    def props(self):
        return ""

    def __repr__(self):
        return f"<Loss:{type(self).__name__}[{self.props}]>"

    def __call__(self, outputs, *args, **kwargs):
        return self._loss(outputs, *args, **kwargs)

    @abstractmethod
    def _loss(self, outputs, *args, **kwargs):
        raise NotImplementedError
