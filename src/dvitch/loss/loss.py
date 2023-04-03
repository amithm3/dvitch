class Loss:
    @property
    def props(self):
        return ""

    def __repr__(self):
        return f"<Loss:{type(self).__name__}[{self.props}]>"

    def __call__(self, outputs, *args, **kwargs):
        return self._loss(outputs, *args, **kwargs)

    def _loss(self, outputs, targets, *args, **kwargs):
        raise NotImplementedError
