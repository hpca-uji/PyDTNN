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

from abc import ABC


class LRScheduler(ABC):
    """
    LRScheduler base class
    """

    def __init__(self, model, verbose):
        self.model = model
        self.verbose = verbose
        self.epoch_count = 0

    def __str__(self):
        return f"LRScheduler {type(self).__name__}"

    def on_batch_begin(self, *args):
        pass

    def on_batch_end(self, *args):
        pass

    def on_epoch_begin(self, *args):
        pass

    def on_epoch_end(self, *args):
        pass

    def log(self, text):
        if self.verbose and self.model.rank == 0:
            print(f"{self}: {text}")
