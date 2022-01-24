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

import numpy as np

from . import LRSchedulerWithLossOrMetric


class ReduceLROnPlateau(LRSchedulerWithLossOrMetric):
    """
    ReduceLROnPlateau LRScheduler
    """

    def __init__(self, model, loss_or_metric="", factor=0.1, patience=5, min_lr=0, verbose=True):
        super().__init__(model, loss_or_metric, verbose)
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.best_epoch = 0
        self.best_loss = np.inf * {True: -1, False: 1}["accuracy" in self.loss_or_metric]

    def on_epoch_end(self, train_loss, val_loss):
        idx = self._get_idx()
        self.epoch_count += 1
        loss = val_loss if self.is_val_metric else train_loss
        if ("accuracy" in self.loss_or_metric and loss[idx] > self.best_loss) or \
                ("accuracy" not in self.loss_or_metric and loss[idx] < self.best_loss):
            self.best_loss = loss[idx]
            self.best_epoch = self.epoch_count
        elif self.epoch_count - self.best_epoch >= self.patience \
                and self.model.optimizer.learning_rate * self.factor >= self.min_lr:
            self.model.optimizer.learning_rate *= self.factor
            self.best_epoch = self.epoch_count
            self.log("Metric '{}' did not improve for {} epochs, setting learning rate to {:.8f}."
                     .format(self.loss_or_metric, self.patience, self.model.optimizer.learning_rate))
