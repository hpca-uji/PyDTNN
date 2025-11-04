import numpy as _np
type _npDT = _np.int8 | _np.float32 | _np.float64
type _npDT_4Dims[T] = _np.ndarray[tuple[int, int, int, int], T]
type _npDT_3Dims[T] = _np.ndarray[tuple[int, int, int], T]
type _npDT_2Dims[T] = _np.ndarray[tuple[int, int], T]
type _npDT_1Dims[T] = _np.ndarray[tuple[int], T]


def div_arrays_set_if_zero[T: _npDT](dividend: _npDT_1Dims[T],
                                     divider: _npDT_1Dims[T],
                                     default_value: T):  # type: ignore
    """
    This function execute a element wise division between dividend and divider ("dividend / divider"), but if divider is 0, then the result is 0 too.
    Example:
        dividend = [-1, 1, 0, 10, -29, 3, 0, 0]
        divider = [0, 40, 0, 0, 3, 0, 10, -30]
        default_value = 0

        result = [0, 1/40, 0, 0, -29/3, 0, 0, 0]

    Another example:
        dividend = [-1, 1, 0, 10, -29, 3, 0, 0]
        divider = [0, 40, 0, 0, 3, 0, 10, -30]
        default_value = 33.3

        result = [33.3, 1/40, 33.3, 33.3, -29/3, 33.3, 0, 0]

    Args:
        dividend (npDT_1Dims): The division's dividend.
        divider (npDT_1Dims): The division's divider.
        default_value (npDT): The va

    Returns:
        Nothing. The value is stores in "dividend".
    """
