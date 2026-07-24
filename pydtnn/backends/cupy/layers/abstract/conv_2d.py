"""CuPy implementation of abstract 2D convolution layers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from cupy import ndarray  # pyright: ignore[reportAttributeAccessIssue]
from cupy.cuda import Stream

from pydtnn.backends.cupy.layers.abstract.layer import LayerCupy
from pydtnn.backends.numpy.layers.abstract.conv_2d import AbstractConv2DNumpy
from pydtnn.layers.abstract.conv_2d import AbstractConv2D
from pydtnn.utils.constants import DTYPE2CTYPE, ArrayShape

__all__ = ("AbstractConv2DCupy",)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from numpy import ndarray  # noqa: F811 (override typing)


class AbstractConv2DCupy(AbstractConv2DNumpy, AbstractConv2D[ndarray], LayerCupy):
    """Abstract base class for 2D convolution layers using CuPy backend."""

    def _model_init(self, prev_shape: ArrayShape, x: ndarray) -> None:
        """Initialize model parameters and CUDA kernels."""
        super()._model_init(prev_shape, x)

        self.stream_2 = Stream()

        self._im2row = self._get_kernel(
            func_name="im2_row_col",
            defines_replaces={
                '"TYPE"': DTYPE2CTYPE[self.model.dtype],
                "TENSOR_FORMAT": self.model.tensor_format,
            },
        )
        self._im2col = self._get_kernel(
            func_name="im2_row_col",
            defines_replaces={
                '"TYPE"': DTYPE2CTYPE[self.model.dtype],
                "TENSOR_FORMAT": self.model.tensor_format,
            },
        )
        self._row2im = self._get_kernel(
            func_name="row_col_2im",
            defines_replaces={
                '"TYPE"': DTYPE2CTYPE[self.model.dtype],
                "TENSOR_FORMAT": self.model.tensor_format,
            },
        )
        self._col2im = self._get_kernel(
            func_name="row_col_2im",
            defines_replaces={
                '"TYPE"': DTYPE2CTYPE[self.model.dtype],
                "TENSOR_FORMAT": self.model.tensor_format,
            },
        )

    def im2row(self, x: ndarray, x_rows: ndarray) -> None:
        """Perform im2row transformation on GPU."""
        # return super().im2row(x, x_rows)
        self._im2row(
            self.model.cuda_grid,
            self.model.cuda_block,
            (
                x,
                x_rows,
                x.shape[0],
                self.ci,
                self.hi,
                self.wi,
                self.kh,
                self.kw,
                self.ho,
                self.wo,
                self.hpadding,
                self.wpadding,
                self.hstride,
                self.wstride,
                self.hdilation,
                self.wdilation,
            ),
        )

    def row2im(self, x_rows: ndarray, dx: ndarray) -> None:
        """Perform row2im transformation on GPU."""
        # return super().im2row(x_rows, dx)
        self._row2im(
            self.model.cuda_grid,
            self.model.cuda_block,
            (
                x_rows,
                dx,
                dx.shape[0],
                self.ci,
                self.hi,
                self.wi,
                self.kh,
                self.kw,
                self.ho,
                self.wo,
                self.hpadding,
                self.wpadding,
                self.hstride,
                self.wstride,
                self.hdilation,
                self.wdilation,
            ),
        )

    def im2col(self, x: ndarray, x_cols: ndarray) -> None:
        """Perform im2col transformation on GPU."""
        # return super().im2col(x, x_cols)
        self._im2row(
            self.model.cuda_grid,
            self.model.cuda_block,
            (
                x,
                x_cols,
                x.shape[0],
                self.ci,
                self.hi,
                self.wi,
                self.kh,
                self.kw,
                self.ho,
                self.wo,
                self.hpadding,
                self.wpadding,
                self.hstride,
                self.wstride,
                self.hdilation,
                self.wdilation,
            ),
        )

    def col2im(self, x_cols: ndarray, dx: ndarray) -> None:
        """Perform col2im transformation on GPU."""
        # return super().im2row(x_cols, dx)
        self._col2im(
            self.model.cuda_grid,
            self.model.cuda_block,
            (
                x_cols,
                dx,
                dx.shape[0],
                self.ci,
                self.hi,
                self.wi,
                self.kh,
                self.kw,
                self.ho,
                self.wo,
                self.hpadding,
                self.wpadding,
                self.hstride,
                self.wstride,
                self.hdilation,
                self.wdilation,
            ),
        )
