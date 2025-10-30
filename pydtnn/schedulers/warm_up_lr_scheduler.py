from typing import TYPE_CHECKING

from pydtnn.schedulers.scheduler import Scheduler

from numpy import ndarray

if TYPE_CHECKING:
    from pydtnn.model import Model


class WarmUp(Scheduler):
    """
    WarmUp
    """

    def __init__(self, warmup_epochs=5, base_lr=1e-4, init_lr=1e-3, verbose=False):
        super().__init__(verbose)
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.init_lr = init_lr
        self.epoch_count: int = 0

    def on_epoch_end(self, train_loss: ndarray[float], val_loss: ndarray[float]) -> None:
        if self.epoch_count < self.warmup_epochs:
            self.model.optimizer.learning_rate = self.base_lr + ((self.epoch_count + 1) / self.warmup_epochs) * (self.init_lr - self.base_lr)
            self.epoch_count += 1
            self.log(f"Setting learning rate to {self.model.optimizer.learning_rate:.8f}.")

    @classmethod
    def from_model(cls, model: "Model") -> "WarmUp":
        return WarmUp(model.warm_up_epochs,
                      model.learning_rate / model.nprocs,
                      model.learning_rate)
