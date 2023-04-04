import time

from dvitch import Trainer, Lambda

if __name__ == "__main__":
    model = Lambda(lambda x: x)
    for e, b in Trainer(10, 100, model, notebook=False):
        time.sleep(.01)
