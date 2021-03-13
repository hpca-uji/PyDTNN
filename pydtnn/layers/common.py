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

from ..model import TRAIN_MODE


# @todo: will be used when layer.initialize includes model: initialize(model, id, ...)
class ForwardToBackward:
    """
    Class used to store those items from the forward pass that are required on the backward pass. When the model
    is in evaluate mode, the passed items are not stored.
    """

    def __init__(self):
        self._model = None
        self._storage = {}

    def set_model(self, model):
        self._model = model

    def __setattr__(self, key, value):
        if self._model.mode == TRAIN_MODE:
            self._storage[key] = value
        else:
            if self._storage:
                self._storage.clear()

    def __getattr__(self, item):
        try:
            return self._storage[item]
        except KeyError:
            raise AttributeError from None
