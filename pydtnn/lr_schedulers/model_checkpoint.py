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

import os
import time

import numpy as np

from . import LRSchedulerWithLossOrMetric


class ModelCheckpoint(LRSchedulerWithLossOrMetric):
    """
    ModelCheckpoint LRScheduler
    """

    def __init__(self, model, loss_or_metric="", epoch_save_frequency=1, verbose=True):
        super().__init__(model, loss_or_metric, verbose)
        self.epoch_save_frequency = epoch_save_frequency
        self.epoch_count = self.best_epoch = 0
        self.best_loss = np.inf * {True: -1, False: 1}["accuracy" in self.loss_or_metric]
        # Attributes that will be properly defined elsewhere
        self.filename = None
        self.last_filename = None

    def on_epoch_end(self, train_loss, val_loss):
        idx = self._get_idx()
        self.epoch_count += 1
        loss = val_loss if self.is_val_metric else train_loss
        if ("accuracy" in self.loss_or_metric and loss[idx] > self.best_loss) or \
                ("accuracy" not in self.loss_or_metric and loss[idx] < self.best_loss):
            self.best_loss = loss[idx]
            self.best_epoch = self.epoch_count
            if self.epoch_count % self.epoch_save_frequency == 0:
                self.filename = "./model-{}-epoch-{}-{}.npz" \
                    .format(self.model.model_name, self.epoch_count, time.strftime("%Y%m%d"))
                self.model.store_weights_and_bias(self.filename)
                self.log(f"Saving model weights and bias in '{self.filename}'.")
                if self.model.rank == 0 and self.last_filename is not None:
                    os.remove(self.last_filename)
                self.last_filename = self.filename
