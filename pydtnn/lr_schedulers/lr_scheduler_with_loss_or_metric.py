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


class LRSchedulerWithLossOrMetric(LRScheduler):
    """
    LRScheduler with metric base class
    """

    def __init__(self, model, loss_or_metric, verbose):
        super().__init__(model, verbose)
        self.is_val_metric = "val_" == loss_or_metric[:4]
        self.loss_or_metric = loss_or_metric[4:] if self.is_val_metric else loss_or_metric

    def _get_idx(self):
        try:
            return self.model.loss_and_metrics.index(self.loss_or_metric)
        except ValueError:
            raise SystemExit("{self}: loss or metric '{self.loss_or_metric}' not found in current model!")
