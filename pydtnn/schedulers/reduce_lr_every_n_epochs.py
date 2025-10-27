from pydtnn.schedulers.lr_scheduler import LRScheduler
from numpy import ndarray


class ReduceLREveryNEpochs(LRScheduler):
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
