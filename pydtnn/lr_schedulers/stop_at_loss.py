#
#  This file is part of Python Distributed Training of Neural Networks (PyDTNN)
#
#  Copyright (C) 2021-22 Universitat Jaume I
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

from . import LRSchedulerWithLossOrMetric


class StopAtLoss(LRSchedulerWithLossOrMetric):
    """
    StopAtLoss LRScheduler
    """

    def __init__(self, model, loss_or_metric="", threshold_value=0, verbose=True):
        super().__init__(model, loss_or_metric, verbose)
        self.threshold_value = threshold_value
        self.stop_training = False

    def on_epoch_end(self, train_loss, val_loss):
        idx = self._get_idx()
        self.epoch_count += 1
        loss = val_loss if self.is_val_metric else train_loss
        if ("accuracy" in self.loss_or_metric and loss[idx] > self.threshold_value) or \
                ("accuracy" not in self.loss_or_metric and loss[idx] < self.threshold_value):
            self.stop_training = True
            self.log("Metric '{self.loss_or_metric}' reached threshold value {self.threshold_value}, stop training.")
