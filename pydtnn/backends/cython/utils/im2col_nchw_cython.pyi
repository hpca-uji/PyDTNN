"""
Cython-accelerated utilities for NCHW image-to-column and column-to-image transformations.
"""

from pydtnn.backends.cython.utils.base import _npDT, _npDT_2Dims, _npDT_4Dims

def im2col_nchw_cython[T: _npDT](
    x: _npDT_4Dims[T], cols: _npDT_2Dims[T], kh: int, kw: int, ho: int, wo: int, vpadding: int, hpadding: int, vstride: int, hstride: int, vdilation: int, hdilation: int
) -> None:
    """
    Rearranges an NCHW image tensor into column format for convolution operations.

    Args:
        x (npDT_4Dims): The 4 dimensional array (the image).
        cols (npDT_2Dims): The 2 dimensional array where the image as columns will be stored (it should be initialized as zeros).
        kh (int): Kernel's heigth.
        kw (int): Kernel's width.
        ho (int): Output's heigth.
        wo (int): Output's width.
        vpadding (int): vertical padding value.
        hpadding (int): horizontal padding value.
        vstride (int): vertical stride value.
        hstride (int): horizontal stride value.
        vdilation (int): vertical dilation value.
        hdilation (int): horizontal dilation value.
    Returns:
        Nothing. The output is stored in "cols".
    """

def col2im_nchw_cython[T: _npDT](
    cols: _npDT_2Dims[T],
    dx: _npDT_4Dims[T],
    n: int,
    c: int,
    h: int,
    w: int,
    kh: int,
    kw: int,
    ho: int,
    wo: int,
    vpadding: int,
    hpadding: int,
    vstride: int,
    hstride: int,
    vdilation: int,
    hdilation: int,
) -> None:
    """
    Rearranges column-formatted data back into an NCHW image tensor.

    Args:
        cols (npDT_2Dims): The 2 dimensional array (the image as columns).
        dx (npDT_4Dims): The 4 dimensional array wher the image will be stored (it should be initialized as zeros).
        n (int): number of samples.
        c (int): number of channels.
        h (int): image's height.
        w (int): image's width.
        kh (int): Kernel's heigth.
        kw (int): Kernel's width.
        ho (int): Output's heigth.
        wo (int): Output's width.
        vpadding (int): vertical padding value.
        hpadding (int): horizontal padding value.
        vstride (int): vertical stride value.
        hstride (int): horizontal stride value.
        vdilation (int): vertical dilation value.
        hdilation (int): horizontal dilation value.
    Returns:
        Nothing. The output is stored in "dx".
    """
