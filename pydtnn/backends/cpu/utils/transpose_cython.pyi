import numpy as _np
type _npDT = _np.int8 | _np.float32 | _np.float64
type _npDT_4Dims[T] = _np.ndarray[tuple[int, int, int, int], T]
type _npDT_4Dims[T] = _np.ndarray[tuple[int, int, int], T]
type _npDT_2Dims[T] = _np.ndarray[tuple[int, int], T]
type _npDT_1Dims[T] = _np.ndarray[tuple[int], T]

def transpose_0231_ikj_cython[T](original: _npDT_4Dims[T], transposed: _npDT_4Dims[T]) -> None:
    """
    Transposes a 4D matrix from (0,1,2,3) to (0,2,3,1).
    This is equivalent to transpose a 3D matrix 0x1x2·3 to 0x2·3x1

    Args:
        original npDT_3Dims): The original matrix.
        transposed npDT_3Dims): The matrix to transpose.
    Returns:
        Nothing. The output is stored in "transposed"
    """


def transpose_0231_ijk_cython[T](original: _npDT_4Dims[T], transposed: _npDT_4Dims[T]) -> None:
    """
    Transposes a 4D matrix from (0,1,2,3) to (0,2,3,1).
    This is equivalent to transpose a 3D matrix 0x1x2·3 to 0x2·3x1

    Args:
        original npDT_3Dims): The original matrix.
        transposed npDT_3Dims): The matrix to transpose.
    Returns:
        Nothing. The output is stored in "transposed"
    """

def transpose_0312_ikj_cython[T](original: _npDT_4Dims[T], transposed: _npDT_4Dims[T]) -> None:
    """
    Transposes a 4D matrix from (0,1,2,3) to (0,3,1,2).
    This is equivalent to transpose a 3D matrix 0x1·2x3 to 0x3x1·2

    Args:
        original npDT_3Dims): The original matrix.
        transposed npDT_3Dims): The matrix to transpose.
    Returns:
        Nothing. The output is stored in "transposed"
    """

def transpose_0312_ijk_cython[T](original: _npDT_4Dims[T], transposed: _npDT_4Dims[T]) -> None:
    """
    Transposes a 4D matrix from (0,1,2,3) to (0,3,1,2).
    This is equivalent to transpose a 3D matrix 0x1·2x3 to 0x3x1·2

    Args:
        original npDT_3Dims): The original matrix.
        transposed npDT_3Dims): The matrix to transpose.
    Returns:
        Nothing. The output is stored in "transposed"
    """


def transpose_1023_jik_cython[T](original: _npDT_4Dims[T], transposed: _npDT_4Dims[T]) -> None:
    """
    Transposes a 4D matrix from (0,1,2,3) to (1,0,2,3).
    This is equivalent to transpose a 3D matrix 0x1x2·3 to 1x0x2·3

    Args:
        original npDT_3Dims): The original matrix.
        transposed npDT_3Dims): The matrix to transpose.
    Returns:
        Nothing. The output is stored in "transposed"
    """


def transpose_1023_ijk_cython[T](original: _npDT_4Dims[T], transposed: _npDT_4Dims[T]) -> None:
    """
    Transposes a 4D matrix from (0,1,2,3) to (0,2,3,1).
    This is equivalent to transpose a 3D matrix 0x1x2·3 to 1x0x2·3

    Args:
        original npDT_3Dims): The original matrix.
        transposed npDT_3Dims): The matrix to transpose.
    Returns:
        Nothing. The output is stored in "transposed"
    """
