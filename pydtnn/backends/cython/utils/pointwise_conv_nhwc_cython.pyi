"""
Cython-accelerated pointwise convolution utilities for NCHW data layout.
"""

from pydtnn.backends.cython.utils.base import _npDT, _npDT_3Dims, _npDT_4Dims

def fwd_pointwise_conv_cython_nhwc[T: _npDT](
    x: _npDT_4Dims[T],
    k: _npDT_3Dims[T],
    res: _npDT_4Dims[T],
    vpadding: int,
    hpadding: int,
    vstride: int,
    hstride: int,
) -> None:
    """
    Performs a pointwise convolution in NCHW format using Cython.

    Args:
        x (npDT_4Dims): The 4 dimensional input's ndarray.
        k (npDT_3Dims): The 3dimensions ndarray that contains the kernel.
        out (npDT_4Dims): The 4 dimensional output's ndarray. Must be filled with zeros.
        vpadding (int): vertical padding value.
        hpadding (int): horizontal padding value.
        vstride (int): vertical stride value.
        hstride (int): horizontal stride value.

    Returns:
        Nothing. The value is stored in `out`.
    """

def bwd_pointwise_conv_cython_nhwc[T: _npDT](
    dy: _npDT_4Dims[T],
    x: _npDT_4Dims[T],
    k: _npDT_3Dims[T],
    dx: _npDT_4Dims[T],
    dw: _npDT_3Dims[T],
    vpadding: int,
    hpadding: int,
    vstride: int,
    hstride: int,
) -> None:
    """
    Computes the gradients for pointwise convolution in NCHW format using Cython.

    Args:
        dy (npDT_4Dims): The 4 dimensional array that contains the gradient of the backward's input.
        x (npDT_4Dims): The 4 dimensional array that contains the input forward's.
        k (npDT_3Dims): The 3 dimensional array that contains the kernel.
        dx npDT_4Dims: The 4 dimensional array that contains the input forward's gradient. Must be filled with zeros.
        dw npDT_3Dims: The 3 dimensional array that contains the kernel's gradient
        vpadding (int): vertical padding value.
        hpadding (int): horizontal padding value.
        vstride (int): vertical stride value.
        hstride (int): horizontal stride value.

    Returns:
        Nothing. The outputs are stored in `dx` and `dw`.
    """
