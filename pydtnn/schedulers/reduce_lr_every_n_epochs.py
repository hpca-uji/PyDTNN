from typing import TYPE_CHECKING

from numpy import ndarray

from pydtnn.schedulers.scheduler import Scheduler

if TYPE_CHECKING:
    from pydtnn.model import Model


class ReduceLREveryNEpochs(Scheduler):
    """
    ReduceLREveryNEpochs LRScheduler
    """

    def __init__(self, factor=0.1, nepochs=5, min_lr=0, verbose=True):
        super().__init__(verbose)
        self.factor = factor
        self.nepochs = nepochs
        self.min_lr = min_lr

    def on_epoch_end(self, train_loss: ndarray[float], val_loss: ndarray[float]) -> None:
        self.epoch_count += 1
        if self.epoch_count % self.nepochs == 0 and self.model.optimizer.learning_rate * self.factor >= self.min_lr:
            self.model.optimizer.learning_rate *= self.factor
            self.log(f"Setting learning rate to {self.model.optimizer.learning_rate:.8f}!")

    @classmethod
    def from_model(cls, model: "Model") -> "ReduceLREveryNEpochs":
        return ReduceLREveryNEpochs(model.reduce_lr_every_nepochs_factor,
                                    model.reduce_lr_every_nepochs_nepochs,
                                    model.reduce_lr_every_nepochs_min_lr)
