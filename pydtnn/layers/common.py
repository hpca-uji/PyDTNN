from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pydtnn.model import Model

# @todo: will be used when layer.initialize includes model: initialize(model, id, ...)


class ForwardToBackward:
    """
    Class used to store those items from the forward pass that are required on the backward pass. When the model
    is in evaluate mode, the passed items are not stored.
    """

    def __init__(self):
        self._model: Model = None
        self._storage = {}

    def set_model(self, model: "Model"):
        self._model: Model = model

    def __setattr__(self, key, value):
        if self._model.mode == Model.Mode.TRAIN:
            self._storage[key] = value
        else:
            if self._storage:
                self._storage.clear()

    def __getattr__(self, item):
        try:
            return self._storage[item]
        except KeyError:
            raise AttributeError from None
