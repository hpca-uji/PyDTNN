#
#  This file is part of Python Distributed Training of Neural Networks (PyDTNN)
#
#  Copyright (C) 2021 Universitat Jaume I
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


class ReduceLREveryNEpochs(LRScheduler):
    """
    ReduceLREveryNEpochs LRScheduler
    """

    def __init__(self, factor=0.1, nepochs=5, min_lr=0, verbose=True):
        super().__init__()
        self.factor = factor
        self.nepochs = nepochs
        self.min_lr = min_lr
        self.epoch_count = 0
        self.verbose = verbose

    def on_epoch_end(self, model, optimizer, loss_metrics, train_loss, val_loss, rank):
        self.epoch_count += 1
        if self.epoch_count % self.nepochs == 0 and optimizer.learning_rate * self.factor >= self.min_lr:
            optimizer.learning_rate *= self.factor
            if self.verbose and rank == 0:
                print("LRScheduler %s: setting learning rate to %.8f" % (type(self).__name__, optimizer.learning_rate))
