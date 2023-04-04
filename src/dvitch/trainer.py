from tqdm import tqdm

from . import Module


class Trainer:
    ebar: "tqdm"
    bbar: "tqdm"

    def __init__(
            self,
            epochs: int,
            batches: int,
            model: "Module",
            *,
            notebook: bool = None,
    ):
        if notebook is None:
            try:
                _ = get_ipython  # type: ignore
                notebook = True
            except NameError:
                notebook = False
        self.epochs = epochs
        self.batches = batches
        self.model = model

        if notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange
        self.trange = trange
        self.tqdm = tqdm

    def __iter__(self):
        with self:
            for e in self.ebar:
                self.bbar.reset()
                self.bbar.last_print = 0
                for b in range(self.batches):
                    self.bbar.update(1)
                    yield e, b
                self.bbar.refresh()
                postfix = self.bbar.postfix
                self.ebar.set_postfix_str(postfix if postfix is not None else "")

    def __enter__(self):
        self.ebar = self.trange(self.epochs, desc="Epochs", unit="epoch",
                                position=0, leave=True)
        self.bbar = self.trange(self.batches, desc="Batches", unit="batch",
                                position=1, leave=True)
        self.ebar.__enter__()
        self.bbar.__enter__()
        self.model.training = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.model.training = False
        self.bbar.__exit__(exc_type, exc_val, exc_tb)
        del self.bbar
        self.ebar.__exit__(exc_type, exc_val, exc_tb)
        del self.ebar
