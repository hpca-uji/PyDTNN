import numpy as _np
type _npDT = _np.int8 | _np.float32 | _np.float64
type _npDT_4Dims[T] = _np.ndarray[tuple[int, int, int, int], T]
type _npDT_3Dims[T] = _np.ndarray[tuple[int, int, int], T]
type _npDT_2Dims[T] = _np.ndarray[tuple[int, int], T]
type _npDT_1Dims[T] = _np.ndarray[tuple[int], T]


def average_pool_2d_fwd_nchw_cython[T: _npDT](x: _npDT_4Dims[T], y: _npDT_4Dims[T],
                                              kh: int, kw: int, ho: int, wo: int,
                                              vpadding: int, hpadding: int,
                                              vstride: int, hstride: int,
                                              vdilation: int, hdilation: int) -> None:
    """
    Args:
        x (npDT_4Dims): The 4 dimensional input's ndarray.
        y (npDT_4Dims): The 4 dimensional output's ndarray. (the output's data is stored in this parameter).
        kh (int): The kernel's height.
        kw (int): The kernel's width.
        ho (int): The output's height.
        wo (int): The output's width.
        vpadding (int): The vertical padding value.
        hpadding (int): The horizontal padding value.
        vstride (int): The vertical stride value.
        hstride (int): The horizontal stride value.
        vdilation (int): The vertical dilation value.
        hdilation (int): The horizontal dilation value.

    Returns:
        Nothing. Implictily the output is stored in "y".
    """


def average_pool_2d_bwd_nchw_cython[T: _npDT](dy: _npDT_4Dims[T],
                                              dx: _npDT_4Dims[T],
                                              n: int, h: int, w: int, c: int,
                                              kh: int, kw: int, ho: int, wo: int,
                                              vpadding: int, hpadding: int,
                                              vstride: int, hstride: int,
                                              vdilation: int, hdilation: int) -> None:
    """
    Args:
        dy (npDT_4Dims): The 4 dimensional input's ndarray.
        dx (npDT_4Dims): The 4 dimensional output's ndarray. (the output's data will be stored in this parameter). Note: All values in this parameter should be 0.
        n (int): The number of images (usually, the batch size).
        h (int): The images' height.
        w (int): The images' width.
        c (int): The images' number of channel's(e.g.: RGB = 3 channels).
        kh (int): The kernel's height.
        kw (int): The kernel's width.
        ho (int): The output's height.
        wo (int): The output's width.
        vpadding (int): The vertical padding value.
        hpadding (int): The horizontal padding value.
        vstride (int): The vertical stride value.
        hstride (int): The horizontal stride value.
        vdilation (int): The vertical dilation value.
        hdilation (int): The horizontal dilation value.

    Returns:
        Nothing. The output will be stored in "dx".
    """
