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


class StopAtLoss(LRScheduler):
    """
    StopAtLoss LRScheduler
    """

    def __init__(self, loss_metric="", threshold_value=0, verbose=True):
        super().__init__()
        self.loss_metric = loss_metric
        self.is_val_metric = "val_" in self.loss_metric
        check_val = self.loss_metric.split("_")
        if "val" == check_val[0]:
            self.loss_metric_ = "_".join(check_val[1:])
        self.threshold_value = threshold_value
        self.stop_training = False
        self.epoch_count = 0
        self.verbose = verbose

    def on_epoch_end(self, model, optimizer, loss_metrics, train_loss, val_loss, rank):
        try:
            idx = loss_metrics.index(self.loss_metric_)
        except:
            idx = 0
        self.epoch_count += 1
        loss = val_loss if self.is_val_metric else train_loss
        if ("accuracy" in self.loss_metric and loss[idx] > self.threshold_value) or \
                ("accuracy" not in self.loss_metric and loss[idx] < self.threshold_value):
            self.stop_training = True
            if self.verbose and rank == 0:
                print("LRScheduler %s: metric %s reached threshold value %f, stop training!" %
                      (type(self).__name__, self.loss_metric, self.threshold_value))
