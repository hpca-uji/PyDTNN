import numpy as _np
from pydtnn.backends.cython.utils.base import _npDT, _npDT_4Dims, _npDT_3Dims, _npDT_2Dims, _npDT_1Dims


def max_pool_2d_fwd_nhwc_cython[T: _npDT](x: _npDT_4Dims[T],
                                          y: _npDT_4Dims[T],
                                          idx_max: _np.ndarray[tuple[int, int, int, int], _np.int32],
                                          kh: int, kw: int, ho: int, wo: int,
                                          vpadding: int, hpadding: int,
                                          vstride: int, hstride: int,
                                          vdilation: int, hdilation: int,
                                          minval: _npDT | int | float) -> None:
    """
    Args:
        x (npDT_4Dims): 4-dimensinal array where the input data is stored.
        y (npDT_4Dims): 4-dimensinal array where the output data will be stored.
        idx_max (np.ndarray[tuple[int, int, int, int], np.int32]): 4-dimensinal array where the index of the maximum values will be stored.
        kh (int): Kernel's heigth.
        kw (int): Kernel's width.
        ho (int): Output's heigth.
        wo (int): Output's width.
        vpadding (int): Vertical padding value.
        hpadding (int): Horizontal padding value.
        vstride (int): Vertical stride value.
        hstride (int): Horizontal stride value.
        vdilation (int): Vertical dilation value.
        hdilation (int): Horizontal dilation value.
        minval (npDT): minum value the selected type can have.
    Returns:
        Nothing. The output is stored in "y" and "idx_max".
    """


def max_pool_2d_bwd_nhwc_cython[T: _npDT](dy: _npDT_4Dims[T],
                                          idx_max: _np.ndarray[tuple[int, int, int, int], _np.int32],
                                          dx: _npDT_4Dims[T],
                                          n: int, h: int, w: int, c: int,
                                          kh: int, kw: int, ho: int, wo: int,
                                          vpadding: int, hpadding: int,
                                          vstride: int, hstride: int,
                                          vdilation: int, hdilation: int) -> None:
    """
    Args:
        dy (npDT_4Dims): 4-dimensinal array where the input data will be stored.
        idx_max (np.ndarray[tuple[int, int, int, int], np.int32]): 4-dimensinal array where the index of the maximum values will be stored.
        dx (npDT_4Dims): 4 dimensional ndarray where the gradient is stored.
        n (int): Number of samples.
        h (int): Sample's heigth.
        w (int): Sample's width.
        c (int): Sample's channels.
        kh (int): Kernel's heigth.
        kw (int): Kernel's width.
        ho (int): Output's heigth.
        wo (int): Output's width.
        vpadding (int): Vertical padding value.
        hpadding (int): Horizontal padding value.
        vstride (int): Vertical stride value.
        hstride (int): Horizontal stride value.
        vdilation (int): Vertical dilation value.
        hdilation (int): Horizontal dilation value.
    Returns:
        Nothing. The output is stored in "dx".
    """
