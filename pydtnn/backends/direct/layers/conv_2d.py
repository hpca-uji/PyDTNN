"""Direct convolution layer implementation using the convDirect library."""

import logging
from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np

from pydtnn.backends.direct.layers.abstract.conv_2d import AbstractConv2DDirect
from pydtnn.backends.numpy.layers.conv_2d import Conv2DNumpy
from pydtnn.libs.convDirect import ConvDirect
from pydtnn.tracers.events import (PYDTNN_EVENT_FINISHED, PYDTNN_OPS_EVENT,
                                   PYDTNN_OPS_EVENTS, OpsEventEnum)
from pydtnn.utils.constants import ArrayShape
from pydtnn.utils.tensor import encode_shape

__all__ = ("Conv2DDirect",)

logger = logging.getLogger(__name__)


class Conv2DDirect(Conv2DNumpy, AbstractConv2DDirect):
    """2D Convolution layer utilizing the direct convolution backend."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the Conv2DDirect layer."""
        super().__init__(*args, **kwargs)
        # convDirect related attributes (will be initialized in initialize())
        self.cd = []

    def _algo_init(self) -> None:
        """Add the different forward and backward methods to the class."""

        def new(name: str, func: Callable) -> None:
            func.__name__ = name
            setattr(self, name, func)

        self._algos = [
            f"convdirect_original_{self.model.tensor_format}_default",
            f"convdirect_renamed_{self.model.tensor_format}_default",
            f"convdirect_reorder_{self.model.tensor_format}_default",
            f"convdirect_block_{self.model.tensor_format}_default",
            f"convdirect_im2row_{self.model.tensor_format}_default",
            f"convdirect_block_blis_{self.model.tensor_format}_blis",
            f"convdirect_conv_gemm_{self.model.tensor_format}_default",
        ]

        for n, method in enumerate(self._algos):
            self.cd.append(
                ConvDirect(
                    method,
                    dtype=self.model.dtype,
                    tensor_format=self.model.tensor_format,
                    debug=self.debug,
                    parent_layer=self,
                )
            )
            new(f"_forward_cd{n}_{self.model.tensor_format}", partial(self._forward_cd, n=n))
            new(f"_backward_cd{n}_{self.model.tensor_format}", partial(self._forward_cd, n=n))

    def _model_init(self, prev_shape: ArrayShape, x: np.ndarray) -> None:
        """Initialize model parameters and select the convolution algorithm."""
        super()._model_init(prev_shape, x)
        self._algo_init()

        if self.model.conv_direct_method:
            try:
                n = self._algos.index(self.model.conv_direct_method)
            except ValueError as e:
                raise ValueError("Specified conv_direct_method not found!") from e
        else:
            n = 0

        self.forward = getattr(self, f"_forward_cd{n}_{self.model.tensor_format}")
        self.backward = getattr(self, f"_backward_cd{n}_{self.model.tensor_format}")

        out_shape = encode_shape((self.model.batch_size, self.co, self.ho, self.wo))
        self.out = np.zeros(out_shape, self.weights.dtype)

        self.out = None

        if self.use_bias:
            logger.warning(f"{self.__class__.__name__} never uses the biases.")

    def _forward_cd(self, x: np.ndarray, n: int = 0) -> np.ndarray:
        """Execute the forward pass using the convDirect library."""

        self.model.tracer.emit_event(
            PYDTNN_OPS_EVENT, self.id * PYDTNN_OPS_EVENTS + OpsEventEnum.FORWARD_CONVDIRECT
        )
        y = self.cd[n].conv_direct(
            np.asarray(self.weights, dtype=self.model.dtype),
            x,
            self.out,
            vpadding=self.hpadding,
            hpadding=self.wpadding,
            vstride=self.hstride,
            hstride=self.wstride,
            vdilation=self.hdilation,
            hdilation=self.wdilation,
        )
        self.model.tracer.emit_event(PYDTNN_OPS_EVENT, PYDTNN_EVENT_FINISHED)
        return y

    def _backward_cd(self, y: np.ndarray, n: int = 0) -> np.ndarray:
        """Execute the backward pass using the convDirect library."""
        raise RuntimeError("Backward not implemented yet!")
