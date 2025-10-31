from pydtnn.backends import PromoteToBackend
from pydtnn.utils import find_component


class Scheduler(PromoteToBackend):
    """
    Scheduler base class
    """

    def __init__(self, verbose: bool):
        self.verbose = verbose
        self.epoch_count = 0
        # NOTE: Only used in early_stopping and stop_at_loss.
        # NOTE (cont.): Since there are only 2 classes that uses this variable,
        #   I think it's not necessary to create an abstract class only to store this variable.
        self.stop_training: bool = False

    def __str__(self):
        return f"Scheduler {type(self).__name__}"

    def on_batch_begin(self, *args):
        pass

    def on_batch_end(self, *args):
        pass

    def on_epoch_begin(self, *args):
        pass

    def on_epoch_end(self, *args):
        pass

    def log(self, text: str):
        if self.verbose and self.model.comm_rank == 0:
            print(f"{self}: {text}")


def select(name: str) -> type[Scheduler]:
    assert __package__, "Package not found!"
    return find_component(__package__, name)
