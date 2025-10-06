from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pydtnn.model import Model
else:
    Model = object

from numpy import ndarray


class LRScheduler(ABC):
    """
    LRScheduler base class
    """

    def __init__(self, model: Model, verbose: bool):
        self.model = model
        self.verbose = verbose
        self.epoch_count = 0
        # NOTE: Only used in early_stopping and stop_at_loss.
        # NOTE (cont.): Since there are only 2 classes that uses this variable,
        #   I think it's not necessary to create an abstract class only to store this variable.
        self.stop_training: bool = False

    def __str__(self):
        return f"LRScheduler {type(self).__name__}"

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

    @abstractmethod
    def on_epoch_end(self, train_loss: ndarray[float], val_loss: ndarray[float]) -> None:
        raise NotImplementedError("\"on_epoch_end\" not imlemented")
