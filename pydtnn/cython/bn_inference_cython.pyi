import numpy as _np
type _npDT = _np.int8 | _np.float32 | _np.float64
type _npDT_4Dims[T] = _np.ndarray[tuple[int, int, int, int], T]
type _npDT_3Dims[T] = _np.ndarray[tuple[int, int, int], T]
type _npDT_2Dims[T] = _np.ndarray[tuple[int, int], T]
type _npDT_1Dims[T] = _np.ndarray[tuple[int], T]

# TODO: Missing `bn_inference_nchw_cython`


def bn_inference_cython[T: _npDT](x: _npDT_2Dims[T],
                                  y: _npDT_2Dims[T],
                                  running_mean: _npDT_1Dims[T],
                                  std: _npDT_1Dims[T],
                                  gamma: _npDT_1Dims[T],
                                  beta: _npDT_1Dims[T]) -> None:
    """
    Args:
        x (npDT_2Dims): The 2 dimensional input's ndarray.
        y (npDT_2Dims): The 2 dimensional outputs's ndarray.
        running_mean (npDT_1Dims): The 1 dimensions ndarray that stores the running mean.
        std (npDT_1Dims): The input's 1 dimensions thtat stores the standard deviation
        gamma (npDT_1Dims): The input's 1 dimensions ndarray the gamma's values
        beta (npDT_1Dims): The input's 1 dimensions ndarray the beta's values

    Returns:
        Nothing. The output is stored in "y".
    """


def bn_relu_inference_cython[T: _npDT](x: _npDT_2Dims[T],
                                       y: _npDT_2Dims[T],
                                       running_mean: _npDT_1Dims[T],
                                       inv_std: _npDT_1Dims[T],
                                       gamma: _npDT_1Dims[T],
                                       beta: _npDT_1Dims[T]) -> None:
    """
    Args:
        x (npDT_2Dims): The 2 dimensional input's ndarray.
        y (npDT_2Dims): The 2 dimensional output's ndarray.
        running_mean (npDT_1Dims): The 1 dimensions ndarray that stores the running mean.
        inv_std (npDT_1Dims): The input's 1 dimensions thtat stores the inverse standard deviation
        gamma (npDT_1Dims): The input's 1 dimensions ndarray the gamma's values
        beta (npDT_1Dims): The input's 1 dimensions ndarray the beta's values

    Returns:
        Nothing. The output will be stored in \"y\".
    """
