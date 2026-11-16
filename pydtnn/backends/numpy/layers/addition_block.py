"""Numpy backend implementation of the AdditionBlock layer."""

import logging
from typing import TYPE_CHECKING

from numpy import ndarray

from pydtnn.backends.numpy.layers.abstract.block_layer import AbstractBlockLayerNumpy
from pydtnn.layers.addition_block import AdditionBlock
from pydtnn.libs import numpy as np
from pydtnn.tracers.events import (PYDTNN_EVENT_FINISHED, PYDTNN_MDL_EVENT, PYDTNN_MDL_EVENTS,
                                   PYDTNN_OPS_EVENT, PYDTNN_OPS_EVENTS, MdlEventEnum, OpsEventEnum)
from pydtnn.utils.constants import ArrayShape

__all__ = ("AdditionBlockNumpy",)


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np  # noqa: F811 (override typing)


class AdditionBlockNumpy(AdditionBlock[np.ndarray], AbstractBlockLayerNumpy):
    """Numpy implementation of an addition block that sums outputs from multiple parallel paths."""

    def _model_init(self, prev_shape: ArrayShape, x: ndarray) -> None:
        super()._model_init(prev_shape, x)

        dx_shape = (len(self.paths), self.model.batch_size, *self.prev_shape)
        self.dx = np.zeros(dx_shape, dtype=self.model.dtype, order="C")
        self.memory_used += self.dx.nbytes

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass by computing the sum of outputs from all parallel paths.

        Args:
            x: Input tensor.

        Returns:
            The element-wise sum of the outputs of all paths.
        """

        num_paths = len(self.paths)
        p = self.paths[0]
        x_forward = x[0] if self.num_inputs > 1 else x
        for layer in p:
            self.model.tracer.emit_event(
                PYDTNN_MDL_EVENT, layer.id * PYDTNN_MDL_EVENTS + MdlEventEnum.FORWARD
            )
            x_forward = layer.forward(x_forward)
            self.model.tracer.emit_event(PYDTNN_MDL_EVENT, PYDTNN_EVENT_FINISHED)
        sum_forwards = x_forward

        for i in range(1, num_paths):
            p = self.paths[i]
            x_forward = x[i] if self.num_inputs > 1 else x
            for layer in p:
                self.model.tracer.emit_event(
                    PYDTNN_MDL_EVENT, layer.id * PYDTNN_MDL_EVENTS + MdlEventEnum.FORWARD
                )
                x_forward = layer.forward(x_forward)
                self.model.tracer.emit_event(PYDTNN_MDL_EVENT, PYDTNN_EVENT_FINISHED)
            self.model.tracer.emit_event(
                PYDTNN_OPS_EVENT,
                self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.FORWARD_ELTW_SUM,
            )
            np.add(sum_forwards, x_forward, out=sum_forwards, dtype=self.model.dtype)
            self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)

        return np.asarray(sum_forwards, dtype=self.model.dtype, order="C")

    def backward(self, dy: np.ndarray) -> np.ndarray:
        """
        Performs the backward pass by propagating gradients through all parallel paths.

        Args:
            dy: Gradient of the loss with respect to the output.

        Returns:
            The accumulated gradient with respect to the input.
        """
        num_paths = len(self.paths)
        real_batch_size = dy.shape[0]
        self.dx.fill(0)

        for i in range(num_paths):
            p = self.paths[i]
            dx = dy
            for layer in reversed(p):
                self.model.tracer.emit_event(
                    PYDTNN_MDL_EVENT, layer.id * PYDTNN_MDL_EVENTS + MdlEventEnum.BACKWARD
                )
                dx = layer.backward(dx)
                self.model.tracer.emit_event(PYDTNN_MDL_EVENT, PYDTNN_EVENT_FINISHED)
            self.dx[i, :real_batch_size, :] = dx

            self.model.tracer.emit_event(
                PYDTNN_OPS_EVENT,
                self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.BACKWARD_ELTW_SUM,
            )

        if self.num_inputs == 1:
            for i in range(num_paths):
                np.add(self.dx[0], self.dx[i], out=self.dx[0], dtype=self.model.dtype)
            output = self.dx[0, :real_batch_size, :]
        else:
            output = self.dx[:, :real_batch_size, :]

        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return np.asarray(output, dtype=self.model.dtype, order="C")
