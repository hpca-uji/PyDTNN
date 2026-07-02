"""Module providing the BestOfVariant class for dynamic convolution implementation selection."""

import logging
from typing import Any
from collections.abc import Callable

import numpy as np

from pydtnn.backends.numpy.layers.conv_2d.direct_cpu import Conv2DDirectNumpy  # type: ignore  # FIXME: too old
from pydtnn.backends.numpy.layers.conv_2d.winograd_cpu import Conv2DWinogradNumpy  # type: ignore  # FIXME: too old
from pydtnn.model import Model
from pydtnn.utils.best_of.best_of import BestOf
from pydtnn.utils.constants import ArrayShape

__all__ = ("BestOfVariant",)

logger = logging.getLogger(__name__)


# FIXME: Broken since Conv2D to backend support
class BestOfVariant(Conv2DWinogradNumpy, Conv2DDirectNumpy):
    """Convolution variant that dynamically selects the most efficient implementation"""

    def __init__(self, *args: Any, **kwargs: dict) -> None:
        """Initializes the BestOfVariant layer with default attributes."""
        super().__init__(*args, **kwargs)
        # best_of related attributes (will be initialized in initialize())
        self._best_fw: BestOf = None  # type: ignore
        self._best_fw_bw_pipeline: BestOf = None  # type: ignore
        # Other parameters
        self.variant = None

    def initialize(self, prev_shape: ArrayShape, x: np.ndarray | None = None) -> None:
        """
        Initializes the layer and configures the BestOf selectors for forward and backward passes.

        Args:
            prev_shape: The shape of the input tensor.
            x: Optional input data for initialization.
        """
        super().initialize(prev_shape, x)
        if self.model.enable_best_of:
            # Set variant to 'best_of' and set alternatives to only forward, and
            # forward backward best_ofs
            self.variant = Conv2DDirectNumpy.Variant.BEST_OF
            # Bestof will honor the next configuration options:
            # - enable_conv_winograd
            # - enable_conv_gemm
            # - enable_conv_direct
            # - conv_direct_methods_for_best_of (if empty, conv_direct_method will be used instead)
            # Set alternatives for only forward, and for forward backward
            alternatives_fw = []
            alternatives_fw_bw_pipeline = []
            if self.model.enable_conv_i2c:
                alternatives_fw.append(
                    (
                        Conv2DDirectNumpy.Variant.I2C,
                        self._get_class_forward_and_backward(Conv2DDirectNumpy.Variant.I2C)[0],
                    )
                )
                alternatives_fw_bw_pipeline.append(
                    (
                        Conv2DDirectNumpy.Variant.I2C,
                        self._get_class_forward_and_backward(Conv2DDirectNumpy.Variant.I2C),
                    )
                )
            if self.model.enable_conv_gemm:
                alternatives_fw.append(
                    (
                        Conv2DDirectNumpy.Variant.GEMM,
                        self._get_class_forward_and_backward(Conv2DDirectNumpy.Variant.GEMM)[0],
                    )
                )
                alternatives_fw_bw_pipeline.append(
                    (
                        Conv2DDirectNumpy.Variant.GEMM,
                        self._get_class_forward_and_backward(Conv2DDirectNumpy.Variant.GEMM),
                    )
                )
            if self.model.enable_conv_winograd and self.cw_constraints_fulfilled:
                alternatives_fw.append(
                    (
                        Conv2DDirectNumpy.Variant.WINOGRAD,
                        self._get_class_forward_and_backward(Conv2DDirectNumpy.Variant.WINOGRAD)[0],
                    )
                )
                alternatives_fw_bw_pipeline.append(
                    (
                        Conv2DDirectNumpy.Variant.WINOGRAD,
                        self._get_class_forward_and_backward(Conv2DDirectNumpy.Variant.WINOGRAD),
                    )
                )
            if self.model.enable_conv_direct:
                for n in range(len(self.cd)):
                    cdn = f"cd{n}"
                    alternatives_fw.append((cdn, self._get_class_forward_and_backward(cdn)[0]))
                    alternatives_fw_bw_pipeline.append(
                        (cdn, self._get_class_forward_and_backward(cdn))
                    )
            self._best_fw = BestOf(
                name="Conv2DNumpy only forward",
                alternatives=alternatives_fw,
                get_problem_size=lambda *args: (
                    tuple(args[0].shape)
                    + tuple(args[0].weights.shape)
                    + (args[0].vstride, args[0].hstride, args[0].vdilation, args[0].hdilation)
                ),
            )
            self._best_fw_bw_pipeline = BestOf(
                name="Conv2DNumpy forward backward",
                alternatives=alternatives_fw_bw_pipeline,
                get_problem_size=lambda *args: (
                    tuple(args[0].shape)
                    + tuple(args[0].weights.shape)
                    + (
                        args[0].vpadding,
                        args[0].hpadding,
                        args[0].vstride,
                        args[0].hstride,
                        args[0].vdilation,
                        args[0].hdilation,
                    )
                ),
            )

    def _get_class_forward_and_backward(self, variant: str) -> list[Callable]:
        """
        Retrieves the forward and backward method references for a given variant.

        Args:
            variant: The identifier of the convolution variant.

        Returns:
            A list containing the forward and backward callable methods.
        """
        return [
            getattr(self.__class__, f"_forward_{variant}_{self.model.tensor_format}"),
            getattr(self.__class__, f"_backward_{variant}_{self.model.tensor_format}"),
        ]

    def _fw_bw_best_of(self, stage: int, x_or_y: np.ndarray) -> BestOf:
        """
        Executes the best performing implementation for the specified stage.

        Args:
            stage: The stage index (0 for forward, 1 for backward).
            x_or_y: The input tensor for forward or gradient tensor for backward.
        """
        match self.model.mode:
            case Model.Mode.TRAIN:
                return self._best_fw_bw_pipeline(stage, self, x_or_y)
            case Model.Mode.EVALUATE:
                return self._best_fw(self, x_or_y)
            case _:
                raise RuntimeError(
                    "Conv2D BestOf variant requires Model.mode to be set to ModelModeEnum.EVALUATE"
                    " or ModelModeEnum.TRAIN"
                )

    def _forward_best_of_nhwc(self, x: np.ndarray) -> BestOf:
        """Performs forward pass using the best variant for NHWC format."""
        return self._fw_bw_best_of(0, x)

    def _forward_best_of_nchw(self, x: np.ndarray) -> BestOf:
        """Performs forward pass using the best variant for NCHW format."""
        return self._fw_bw_best_of(0, x)

    def _backward_best_of_nhwc(self, y: np.ndarray) -> BestOf:
        """Performs backward pass using the best variant for NHWC format."""
        return self._fw_bw_best_of(1, y)

    def _backward_best_of_nchw(self, y: np.ndarray) -> BestOf:
        """Performs backward pass using the best variant for NCHW format."""
        return self._fw_bw_best_of(1, y)
