class Trainer:
    def __init__(self, model, loss_fn):
        self.model = model
        self.loss_fn = loss_fn

    def __enter__(self):
        self.model.training = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.model.training = False

    def __call__(self, *args, **kwargs):
        pass


def train_printer(
        trainer: "Trainer",
        epochs: int,
        batches: int,
        interval: float = 0.5,
        *,
        notebook: bool = False
):
    if notebook:
        from tqdm.notebook import tqdm, trange
    else:
        from tqdm import tqdm, trange

    ebar = trange(epochs, desc="Epochs", unit="epoch", position=0, leave=True)
    bbar = tqdm(total=batches, desc="Batches", unit="batch", position=1, leave=True)

    with bbar:
        for e in ebar:
            bbar.reset()
            bbar.last_print = 0
            for b in range(batches):
                bbar.update(1)
                props = trainer(e, b)
                if (curr := bbar.format_dict["elapsed"]) - bbar.last_print > interval:
                    bbar.last_print = curr
                    bbar.set_postfix(**props)
            bbar.refresh()
            ebar.set_postfix(**props)
