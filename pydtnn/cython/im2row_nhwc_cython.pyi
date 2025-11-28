import numpy as _np
type _npDT = _np.int8 | _np.float32 | _np.float64
type _npDT_4Dims[T] = _np.ndarray[tuple[int, int, int, int], T]
type _npDT_3Dims[T] = _np.ndarray[tuple[int, int, int], T]
type _npDT_2Dims[T] = _np.ndarray[tuple[int, int], T]
type _npDT_1Dims[T] = _np.ndarray[tuple[int], T]


def im2row_nhwc_cython[T: _npDT](x: _npDT_4Dims[T],
                                 rows: _npDT_2Dims[T],
                                 kh: int, kw: int, ho: int, wo: int,
                                 vpadding: int, hpadding: int,
                                 vstride: int, hstride: int,
                                 vdilation: int, hdilation: int) -> None:
    """
    Args:
        x (npDT_4Dims): The 4 dimensional array (the image).
        rows (npDT_2Dims): The 2 dimensional array where the image as columns is stored (it should be initalized with 0s).
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
        Nothing. The output is sotred in "rows".
    """


def row2im_nhwc_cython[T: _npDT](rows: _npDT_2Dims[T],
                                 x: _npDT_4Dims[T],
                                 n: int, h: int, w: int, c: int,
                                 kh: int, kw: int, ho: int, wo: int,
                                 vpadding: int, hpadding: int,
                                 vstride: int, hstride: int,
                                 vdilation: int, hdilation: int) -> None:
    """
    Args:
        rows (npDT_2Dims): The 2 dimensional array (the image).
        x (npDT_4Dims): The 4 dimensional array where the image will be stored (it should be initalized with 0s).
        n (int): number of samples.
        h (int): image's height.
        w (int): image's width.
        c (int): number of channels.
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
        Nothing. The output is stored in "x".
    """
