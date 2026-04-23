from typing import Any
import logging
from warnings import warn

import numpy as np

from pydtnn.context.layer import Layer
from pydtnn.utils.constants import Array, Parameters
logger = logging.getLogger(__name__)


class Export[T: Array](Layer[T]):
    def export(self) -> dict[str, Any]:
        """Export model state"""
        data = {}

        if self.model_name is not None:
            data[Parameters.MODEL_NAME] = self.model_name

        data[Parameters.LAYERS] = [
            layer.export()
            for layer in self.layers
        ]

        return data

    def import_(self, data: "dict[str, Any] | Export") -> None:
        """Import model state"""
        if isinstance(data, Export):
            data = data.export()

        model_name = str(data.get(Parameters.MODEL_NAME))
        if model_name != self.model_name:
            warn_text = f"Importing from different models! (self: {self.model_name}, got: {model_name})"
            logger.warning(warn_text)
            warn(warn_text, RuntimeWarning)

        for layer, data in zip(self.layers, data[Parameters.LAYERS]):
            layer.import_(data)  # type: ignore (It is the right data type.)

    def load_weights_and_bias(self, filename: str) -> None:
        """
        ARGS:
            filename: Path to the file with the weights and biases to load.
        """
        with np.load(filename, allow_pickle=True) as data:
            self.import_(data)

    def store_weights_and_bias(self, filename: str, compress=True) -> None:
        """
        ARGS:
            filename: Path to the file were the weights and biases will be stored.
        """
        save = np.savez_compressed if compress else np.savez
        save(filename, **self.export())
