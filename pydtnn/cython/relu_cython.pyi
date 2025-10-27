import numpy as _np
type _npDT = _np.int8 | _np.float32 | _np.float64
type _npDT_4Dims[T] = _np.ndarray[tuple[int, int, int, int], T]
type _npDT_3Dims[T] = _np.ndarray[tuple[int, int, int], T]
type _npDT_2Dims[T] = _np.ndarray[tuple[int, int], T]
type _npDT_1Dims[T] = _np.ndarray[tuple[int], T]


def relu_cython[T:_npDT](x: _npDT_1Dims[T],
                         max: _npDT_1Dims[T],
                         mask: _np.ndarray[tuple[int], _np.int8]) -> None:
    """
    Args:
        x (npDT_1Dims): 1-dimensional input's array.
        max (npDT_1Dims): 1-dimensional array where the ouput is stored
        mask (np.ndarray[tuple[int], np.int8]): 1-dimensional array where the output's mask is stored.
    Returns:
        Nothing. The output is stored in "max" and "mask".
    """


def capped_relu_cython[T:_npDT](x: _npDT_1Dims[T],
                                max: _npDT_1Dims[T],
                                mask: _np.ndarray[tuple[int], _np.int8],
                                cap: float) -> None:
    """
    ReLU function where the values above "cap"'s value are set as this value.

    Note: if cap is 6, this is a Relu6

    Args:
        x (npDT_1Dims): 1-dimensional input's array.
        max (npDT_1Dims): 1-dimensional array where the ouput is stored
        mask (np.ndarray[tuple[int], np.int8]): 1-dimensional array where the output's mask is stored.
        cap (float): The ReLU's superior limit. Any value in x greater that this parameter will be set to this parameter in the ouput.
    Returns:
        Nothing. The output is stored in "max" and "mask".
    """


def leaky_relu_cython[T:_npDT](x: _npDT_1Dims[T],
                               max: _npDT_1Dims[T],
                               mask: _npDT_1Dims[T],
                               negative_slope: float) -> None:
    """
    Args:
        x (npDT_1Dims): 1-dimensional input's array.
        max (npDT_1Dims): 1-dimensional array where the ouput is stored
        mask (np.ndarray[tuple[int], np.int8]): 1-dimensional array where the output's mask is stored.
        negative_slope (float): The negative value's multiplayer (if is 0, this function acts as a normal ReLU)
    Returns:
        Nothing. The output is stored in "max" and "mask".
    """
