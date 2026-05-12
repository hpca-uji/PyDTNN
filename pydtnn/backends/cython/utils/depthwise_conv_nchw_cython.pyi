"""
Cython-accelerated depthwise convolution utilities for NCHW data format.
"""

from pydtnn.backends.cython.utils.base import _npDT, _npDT_3Dims, _npDT_4Dims

def depthwise_conv_nchw_cython[T: _npDT](
    x: _npDT_4Dims[T], k: _npDT_3Dims[T], res: _npDT_4Dims[T], ho: int, wo: int, vpadding: int, hpadding: int, vstride: int, hstride: int, vdilation: int, hdilation: int
) -> _npDT_4Dims[T]:
    """
    Performs a depthwise convolution in NCHW format using Cython acceleration.

    Args:
        x (npDT_4Dims): The 4 dimensional input's ndarray.
        k (npDT_3Dims): The 3dimensions ndarray that contains the kernel.
        res (npDT_4Dims): The 4 dimensional output's ndarray. Must be filled with zeros.
        ho: (int): Output's height value.
        wo: (int): Output's width value.
        vpadding (int): vertical padding value.
        hpadding (int): horizontal padding value.
        vstride (int): vertical stride value.
        hstride (int): horizontal stride value.
        vdilation (int): vertical dilation value.
        hdilation (int): horizontal dilation value.

    Returns:
        Nothing. The value is stores in "res".
    """

def depthwise_conv_backward_nchw_cython[T: _npDT](
    dy: _npDT_4Dims[T], x: _npDT_4Dims[T], k: _npDT_3Dims[T], dx: _npDT_4Dims[T], dw: _npDT_3Dims[T], vpadding: int, hpadding: int, vstride: int, hstride: int, vdilation: int, hdilation: int
) -> None:
    """
    Computes the gradients for input and kernel in a depthwise convolution.

    Args:
        dy (npDT_4Dims): The 4 dimensional array that contains the gradient of the backward's input.
        x (npDT_4Dims): The 4 dimensional array that contains the input forward's.
        k (npDT_3Dims): The 3 dimensional array that contains the kernel.
        dx npDT_4Dims: The 4 dimensional array that contains the input forward's gradient. Must be filled with zeros.
        dw npDT_3Dims: The 3 dimensional array that contains the kernel's gradient.
        vpadding (int): vertical padding value.
        hpadding (int): horizontal padding value.
        vstride (int): vertical stride value.
        hstride (int): horizontal stride value.
        vdilation (int): vertical dilation value.
        hdilation (int): horizontal dilation value.
    Returns:
        Nothing. The outputs are stored in `dx` and `dw`.
    """
