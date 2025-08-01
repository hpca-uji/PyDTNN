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

from abc import ABC

from . import LRScheduler
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pydtnn.model import Model
else:
    Model = object

class LRSchedulerWithLossOrMetric(LRScheduler, ABC):
    """
    LRScheduler with metric base class
    """

    def __init__(self, model: Model, loss_or_metric:str, verbose:bool):
        # NOTE: loss_or_metric default value is "val_accuracy" in Parser.
        super().__init__(model, verbose)
        self.is_val_metric:bool = "val_" == loss_or_metric[:4]
        self.loss_or_metric = loss_or_metric[4:] if self.is_val_metric else loss_or_metric

    def _get_idx(self):
        try:
            return self.model.loss_and_metrics.index(self.loss_or_metric)
        except ValueError:
            raise SystemExit("{self}: loss or metric '{self.loss_or_metric}' not found in current model!")
