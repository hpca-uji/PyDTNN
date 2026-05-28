"""
Utility module for element-wise division with zero-handling logic.
"""

from pydtnn.backends.cython.utils.base import _npDT, _npDT_1Dims

def div_arrays_set_if_zero[T: _npDT](
    dividend: _npDT_1Dims[T], divider: _npDT_1Dims[T], default_value: T
):  # type: ignore
    """
    Performs element-wise division of dividend by divider, replacing results with a default value where the divider is zero.

    The operation is performed in-place on the dividend array.

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
        dividend: The array to be divided, modified in-place.
        divider: The array used as the divisor.
        default_value: The value to assign when the corresponding divider element is zero.

    Returns:
        None
    """
