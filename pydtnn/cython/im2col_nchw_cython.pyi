import numpy as _np
type _npDT = _np.int8 | _np.float32 | _np.float64
type _npDT_4Dims[T] = _np.ndarray[tuple[int, int, int, int], T]
type _npDT_3Dims[T] = _np.ndarray[tuple[int, int, int], T]
type _npDT_2Dims[T] = _np.ndarray[tuple[int, int], T]
type _npDT_1Dims[T] = _np.ndarray[tuple[int], T]


def im2col_nchw_cython[T: _npDT](x: _npDT_4Dims[T],
                                 kh: int, kw: int,
                                 vpadding: int, hpadding: int,
                                 vstride: int, hstride: int,
                                 vdilation: int, hdilation: int) -> None:
    """
    Args:
        x (npDT_4Dims): The 4 dimensional array (the image).
        cols (npDT_2Dims): The 2 dimensional array where the image as columns will be stored (it should be initialized as zeros).
        kh (int): Kernel's heigth.
        kw (int): Kernel's width.
        vpadding (int): vertical padding value.
        hpadding (int): horizontal padding value.
        vstride (int): vertical stride value.
        hstride (int): horizontal stride value.
        vdilation (int): vertical dilation value.
        hdilation (int): horizontal dilation value.
    Returns:
        Nothing. The output is stored in "cols".
    """


def col2im_nchw_cython[T: _npDT](cols: _npDT_2Dims[T],
                                 x: _npDT_4Dims[T],
                                 n: int, h: int, w: int, c: int,
                                 kh: int, kw: int,
                                 vpadding: int, hpadding: int,
                                 vstride: int, hstride: int,
                                 vdilation: int, hdilation: int) -> None:
    """
    Args:
        cols (npDT_2Dims): The 2 dimensional array (the image as columns).
        x (npDT_4Dims): The 4 dimensional array wher the image will be stored (it should be initialized as zeros).
        kh (int): Kernel's heigth.
        kw (int): Kernel's width.
        vpadding (int): vertical padding value.
        hpadding (int): horizontal padding value.
        vstride (int): vertical stride value.
        hstride (int): horizontal stride value.
        vdilation (int): vertical dilation value.
        hdilation (int): horizontal dilation value.
    Returns:
        Nothing. The output is stored in "x".
    """
