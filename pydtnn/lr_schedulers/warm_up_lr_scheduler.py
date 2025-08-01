#
#  This file is part of Python Distributed Training of Neural Networks (PyDTNN)
#
#  Copyright (C) 2021-25 Universitat Jaume I
#
#  PyDTNN is free software: you can redistribute it and/or modify it under the
#  terms of the GNU General Public License as published by the Free Software
#  Foundation, either version 3 of the License, or (at your option) any later
#  version.
#
#  This program is distributed in the hope that it will be useful, but WITHOUT
#  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
#  or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
#  License for more details.
#
#  You should have received a copy of the GNU General Public License along
#  with this program.  If not, see <https://www.gnu.org/licenses/>.
#

from . import LRScheduler
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pydtnn.model import Model
else:
    Model = object

from numpy import ndarray

class WarmUpLRScheduler(LRScheduler):
    """
    WarmUpLRScheduler
    """

    def __init__(self, model:Model, warmup_epochs=5, base_lr=1e-4, init_lr=1e-3, verbose=False):
        super().__init__(model, verbose)
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.init_lr = init_lr
        self.epoch_count:int = 0

    def on_epoch_end(self, train_loss: ndarray[float], val_loss: ndarray[float]) -> None:
        if self.epoch_count < self.warmup_epochs:
            self.model.optimizer.learning_rate = self.base_lr + ((self.epoch_count + 1) / self.warmup_epochs) * (self.init_lr - self.base_lr)
            self.epoch_count += 1
            self.log(f"Setting learning rate to {self.model.optimizer.learning_rate:.8f}.")
