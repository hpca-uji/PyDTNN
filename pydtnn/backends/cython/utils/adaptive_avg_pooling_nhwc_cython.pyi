"""Cython-accelerated adaptive average pooling utilities for NHWC tensor layouts."""

from pydtnn.backends.cython.utils.base import _npDT, _npDT_4Dims

def adaptive_avg_pooling_fwd_nhwc_cython[T: _npDT](  # noqa: D103,E302
    x: _npDT_4Dims[T], pooled_x: _npDT_4Dims[T]
) -> None:
    """
    Performs forward adaptive average pooling on NHWC input data.

    Args:
        x (npDT_4Dims): data input.
        pooled_x (npDT_4Dims): ndarray where the output will be stored.
    Returns:
        Nothing; the return is stored in "dx".
    """

def adaptive_avg_pooling_bwd_nhwc_cython[T: _npDT](dx: _npDT_4Dims[T], dy: _npDT_4Dims[T]) -> None:  # noqa: D103,E302
    """
    Performs backward adaptive average pooling on NHWC input data.

    Args:
        dx (npDT_4Dims): ndarray where the output will be stored.
        dy (npDT_4Dims): data input.
    Returns:
        Nothing; the return is stored in "dx".
    """
