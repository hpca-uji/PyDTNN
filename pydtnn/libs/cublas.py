
#!/usr/bin/env python
# Source: https://github.com/lebedov/scikit-cuda

"""
Python interface to CUBLAS functions.

Note: this module does not explicitly depend on PyCUDA.
"""

from __future__ import absolute_import

import ctypes
import ctypes.util
import re
import sys
from string import Template

import numpy as np

from pydtnn.libs import cuda, utils

# Load library:
__all__ = (
    "cublasAllocFailed",
    "cublasArchMismatch",
    "cublasCaxpy",
    "cublasCcopy",
    "cublasCdgmm",
    "cublasCdotc",
    "cublasCdotu",
    "cublasCgbmv",
    "cublasCgeam",
    "cublasCgelsBatched",
    "cublasCgemm",
    "cublasCgemmBatched",
    "cublasCgemmStridedBatched",
    "cublasCgemv",
    "cublasCgerc",
    "cublasCgeru",
    "cublasCgetrfBatched",
    "cublasCgetriBatched",
    "cublasCgetrsBatched",
    "cublasChbmv",
    "cublasCheckStatus",
    "cublasChemm",
    "cublasChemv",
    "cublasCher",
    "cublasCher2",
    "cublasCher2k",
    "cublasCherk",
    "cublasChpmv",
    "cublasChpr",
    "cublasChpr2",
    "cublasCreate",
    "cublasCrot",
    "cublasCrotg",
    "cublasCscal",
    "cublasCsrot",
    "cublasCsscal",
    "cublasCswap",
    "cublasCsymm",
    "cublasCsymv",
    "cublasCsyr",
    "cublasCsyr2",
    "cublasCsyr2k",
    "cublasCsyrk",
    "cublasCtbmv",
    "cublasCtbsv",
    "cublasCtpmv",
    "cublasCtpsv",
    "cublasCtrmm",
    "cublasCtrmv",
    "cublasCtrsm",
    "cublasCtrsv",
    "cublasDasum",
    "cublasDaxpy",
    "cublasDcopy",
    "cublasDdgmm",
    "cublasDdot",
    "cublasDestroy",
    "cublasDgbmv",
    "cublasDgeam",
    "cublasDgelsBatched",
    "cublasDgemm",
    "cublasDgemmBatched",
    "cublasDgemmStridedBatched",
    "cublasDgemv",
    "cublasDger",
    "cublasDgetrfBatched",
    "cublasDgetriBatched",
    "cublasDgetrsBatched",
    "cublasDnrm2",
    "cublasDrot",
    "cublasDrotg",
    "cublasDrotm",
    "cublasDrotmg",
    "cublasDsbmv",
    "cublasDscal",
    "cublasDspmv",
    "cublasDspr",
    "cublasDspr2",
    "cublasDswap",
    "cublasDsymm",
    "cublasDsymv",
    "cublasDsyr",
    "cublasDsyr2",
    "cublasDsyr2k",
    "cublasDsyrk",
    "cublasDtbmv",
    "cublasDtbsv",
    "cublasDtpmv",
    "cublasDtpsv",
    "cublasDtrmm",
    "cublasDtrmv",
    "cublasDtrsm",
    "cublasDtrsmBatched",
    "cublasDtrsv",
    "cublasDzasum",
    "cublasDznrm2",
    "cublasError",
    "cublasExecutionFailed",
    "cublasGetPointerMode",
    "cublasGetStream",
    "cublasGetVersion",
    "cublasIcamax",
    "cublasIcamin",
    "cublasIdamax",
    "cublasIdamin",
    "cublasInternalError",
    "cublasInvalidValue",
    "cublasIsamax",
    "cublasIsamin",
    "cublasIzamax",
    "cublasIzamin",
    "cublasLicenseError",
    "cublasMappingError",
    "cublasNotInitialized",
    "cublasNotSupported",
    "cublasSasum",
    "cublasSaxpy",
    "cublasScasum",
    "cublasScnrm2",
    "cublasScopy",
    "cublasSdgmm",
    "cublasSdot",
    "cublasSetPointerMode",
    "cublasSetStream",
    "cublasSgbmv",
    "cublasSgeam",
    "cublasSgelsBatched",
    "cublasSgemm",
    "cublasSgemmBatched",
    "cublasSgemmStridedBatched",
    "cublasSgemv",
    "cublasSger",
    "cublasSgetrfBatched",
    "cublasSgetriBatched",
    "cublasSgetrsBatched",
    "cublasSnrm2",
    "cublasSrot",
    "cublasSrotg",
    "cublasSrotm",
    "cublasSrotmg",
    "cublasSsbmv",
    "cublasSscal",
    "cublasSspmv",
    "cublasSspr",
    "cublasSspr2",
    "cublasSswap",
    "cublasSsymm",
    "cublasSsymv",
    "cublasSsyr",
    "cublasSsyr2",
    "cublasSsyr2k",
    "cublasSsyrk",
    "cublasStbmv",
    "cublasStbsv",
    "cublasStpmv",
    "cublasStpsv",
    "cublasStrmm",
    "cublasStrmv",
    "cublasStrsm",
    "cublasStrsmBatched",
    "cublasStrsv",
    "cublasZaxpy",
    "cublasZcopy",
    "cublasZdgmm",
    "cublasZdotc",
    "cublasZdotu",
    "cublasZdrot",
    "cublasZdscal",
    "cublasZgbmv",
    "cublasZgeam",
    "cublasZgelsBatched",
    "cublasZgemm",
    "cublasZgemmBatched",
    "cublasZgemmStridedBatched",
    "cublasZgemv",
    "cublasZgerc",
    "cublasZgeru",
    "cublasZgetrfBatched",
    "cublasZgetriBatched",
    "cublasZgetrsBatched",
    "cublasZhbmv",
    "cublasZhemm",
    "cublasZhemv",
    "cublasZher",
    "cublasZher2",
    "cublasZher2k",
    "cublasZherk",
    "cublasZhpmv",
    "cublasZhpr",
    "cublasZhpr2",
    "cublasZrot",
    "cublasZrotg",
    "cublasZscal",
    "cublasZswap",
    "cublasZsymm",
    "cublasZsymv",
    "cublasZsyr",
    "cublasZsyr2",
    "cublasZsyr2k",
    "cublasZsyrk",
    "cublasZtbmv",
    "cublasZtbsv",
    "cublasZtpmv",
    "cublasZtpsv",
    "cublasZtrmm",
    "cublasZtrmv",
    "cublasZtrsm",
    "cublasZtrsv",
)

_linux_version_list = [11.0, 10.2, 10.1, 10.0, 9.2, 9.1, 9.0, 8.0, 7.5, 7.0, 6.5, 6.0, 5.5, 5.0, 4.0]
_win32_version_list = [11, 10, 10, 100, 92, 91, 90, 80, 75, 70, 65, 60, 55, 50, 40]
if "linux" in sys.platform:
    _libcublas_libname_list = ["libcublas.so"] + ["libcublas.so.%s" % v for v in _linux_version_list]
elif sys.platform == "darwin":
    _libcublas_libname_list = ["libcublas.dylib"]
elif sys.platform == "win32":
    if sys.maxsize > 2**32:
        _libcublas_libname_list = ["cublas.dll"] + ["cublas64_%s.dll" % v for v in _win32_version_list]
    else:
        _libcublas_libname_list = ["cublas.dll"] + ["cublas32_%s.dll" % v for v in _win32_version_list]
else:
    raise RuntimeError("unsupported platform")

# Print understandable error message when library cannot be found:
_libcublas = None
for _libcublas_libname in _libcublas_libname_list:
    try:
        if sys.platform == "win32":
            _libcublas = ctypes.windll.LoadLibrary(_libcublas_libname)
        else:
            _libcublas = ctypes.cdll.LoadLibrary(_libcublas_libname)
    except OSError:
        pass
    else:
        break
if _libcublas is None:
    raise OSError("cublas library not found")

# Generic CUBLAS error:


class cublasError(Exception):
    """Base class for CUBLAS errors."""

    pass


# Exceptions corresponding to different CUBLAS errors:


class cublasNotInitialized(cublasError):
    """CUBLAS library not initialized."""

    pass


class cublasAllocFailed(cublasError):
    """Resource allocation failed."""

    pass


class cublasInvalidValue(cublasError):
    """Unsupported numerical value was passed to function."""

    pass


class cublasArchMismatch(cublasError):
    """Function requires an architectural feature absent from the device."""

    pass


class cublasMappingError(cublasError):
    """Access to GPU memory space failed."""

    pass


class cublasExecutionFailed(cublasError):
    """GPU program failed to execute."""

    pass


class cublasInternalError(cublasError):
    """An internal CUBLAS operation failed."""

    pass


class cublasNotSupported(cublasError):
    """Not supported."""

    pass


class cublasLicenseError(cublasError):
    """License error."""

    pass


cublasExceptions = {
    1: cublasNotInitialized,
    3: cublasAllocFailed,
    7: cublasInvalidValue,
    8: cublasArchMismatch,
    11: cublasMappingError,
    13: cublasExecutionFailed,
    14: cublasInternalError,
    15: cublasNotSupported,
    16: cublasLicenseError,
}

_CUBLAS_OP = {
    0: 0,  # CUBLAS_OP_N
    "n": 0,
    "N": 0,
    1: 1,  # CUBLAS_OP_T
    "t": 1,
    "T": 1,
    2: 2,  # CUBLAS_OP_C
    "c": 2,
    "C": 2,
}

_CUBLAS_FILL_MODE = {
    0: 0,  # CUBLAS_FILL_MODE_LOWER
    "l": 0,
    "L": 0,
    1: 1,  # CUBLAS_FILL_MODE_UPPER
    "u": 1,
    "U": 1,
}

_CUBLAS_DIAG = {
    0: 0,  # CUBLAS_DIAG_NON_UNIT,
    "n": 0,
    "N": 0,
    1: 1,  # CUBLAS_DIAG_UNIT
    "u": 1,
    "U": 1,
}

_CUBLAS_SIDE_MODE = {
    0: 0,  # CUBLAS_SIDE_LEFT
    "l": 0,
    "L": 0,
    1: 1,  # CUBLAS_SIDE_RIGHT
    "r": 1,
    "R": 1,
}


class _types:
    """Some alias types for CUBLAS arguments."""

    handle = ctypes.c_void_p
    stream = ctypes.c_void_p


def cublasCheckStatus(status):
    """
    Raise CUBLAS exception based on status code.

    Raises an exception corresponding to the specified CUBLAS error
    code. If the status code is not recognized, a generic `cublasError`
    is raised.

    Parameters
    ----------
    status : int
        CUBLAS error code returned by a CUBLAS function.

    Raises
    ------
    cublasError
        If the status code indicates an error.

    See Also
    --------
    cublasExceptions
    """

    if status != 0:
        try:
            e = cublasExceptions[status]
        except KeyError:
            raise cublasError("Unknown CUBLAS error code: {}".format(status))
        else:
            raise e


# Helper functions:
_libcublas.cublasCreate_v2.restype = int
_libcublas.cublasCreate_v2.argtypes = [_types.handle]


def cublasCreate():
    """
    Initialize CUBLAS and create a handle.

    Initializes the CUBLAS library and creates a handle to a structure
    that holds the CUBLAS library context. This handle is required for
    most subsequent CUBLAS function calls.

    Returns
    -------
    handle : int
        A CUBLAS context handle.

    References
    ----------
    `cublasCreate <http://docs.nvidia.com/cuda/cublas/#cublascreate>`_
    """

    handle = _types.handle()
    assert _libcublas
    status = _libcublas.cublasCreate_v2(ctypes.byref(handle))
    cublasCheckStatus(status)
    return handle.value


_libcublas.cublasDestroy_v2.restype = int
_libcublas.cublasDestroy_v2.argtypes = [_types.handle]


def cublasDestroy(handle):
    """
    Destroy a CUBLAS handle and release resources.

    Releases hardware resources used by CUBLAS and destroys the
    associated context.

    Parameters
    ----------
    handle : int
        The CUBLAS context handle to destroy.

    References
    ----------
    `cublasDestroy <http://docs.nvidia.com/cuda/cublas/#cublasdestroy>`_
    """

    assert _libcublas
    status = _libcublas.cublasDestroy_v2(handle)
    cublasCheckStatus(status)


_libcublas.cublasGetVersion_v2.restype = int
_libcublas.cublasGetVersion_v2.argtypes = [_types.handle, ctypes.c_void_p]


def cublasGetVersion(handle):
    """
    Get the CUBLAS library version.

    Returns the version number of the installed CUBLAS library.

    Parameters
    ----------
    handle : int
        The CUBLAS context handle.

    Returns
    -------
    version : int
        The CUBLAS version number (e.g., 11000 for 11.0).

    References
    ----------
    `cublasGetVersion <http://docs.nvidia.com/cuda/cublas/#cublasgetversion>`_
    """

    version = ctypes.c_int()
    assert _libcublas
    status = _libcublas.cublasGetVersion_v2(handle, ctypes.byref(version))
    cublasCheckStatus(status)
    return version.value


def _get_cublas_version():
    """
    Get and save CUBLAS version using the CUBLAS library's SONAME.

    This function attempts to determine the CUBLAS version by parsing the
    library's SONAME (Shared Object Name) to avoid creating a CUBLAS context
    if possible, as context creation can sometimes affect performance. If
    parsing fails or is not applicable (e.g., on macOS), it falls back to
    calling `cublasGetVersion`.

    Returns
    -------
    version : str
        The CUBLAS version as a string, formatted with trailing zeros
        (e.g., '6050' for version 6.5).

    Notes
    -----
    On macOS, the SONAME parsing might not be reliable, so `cublasGetVersion`
    is used as a fallback.
    """

    cublas_path = utils.find_lib_path("cublas")
    try:
        match = re.search(r"[\D\.]+\.+(\d+)\.(\d+)", utils.get_soname(cublas_path))
        assert match
        major, minor = match.groups()
    except BaseException:
        # Create a temporary context to run cublasGetVersion():
        # warnings.warn('creating CUBLAS context to get version number')
        h = cublasCreate()
        version = cublasGetVersion(h)
        cublasDestroy(h)
        return str(version)
    else:
        return major.ljust(len(major) + 1, "0") + minor.ljust(2, "0")


_cublas_version = int(_get_cublas_version())


class _cublas_version_req(object):
    """
    Decorator to conditionally enable functions based on CUBLAS version.

    This decorator replaces a decorated function with a placeholder that
    raises `NotImplementedError` if the installed CUBLAS version is less
    than the required version `v`. Otherwise, it returns the original function.
    """

    def __init__(self, v):
        """
        Initialize the version requirement.

        Parameters
        ----------
        v : float or int
            The minimum required CUBLAS version (e.g., 5.0 or 5000).
        """
        self.vs = str(v)
        if isinstance(v, int):
            major = str(v)
            minor = "0"
        else:
            match = re.search(r"(\d+)\.(\d+)", self.vs)
            assert match
            major, minor = match.groups()
        self.vi = major.ljust(len(major) + 1, "0") + minor.ljust(2, "0")

    def __call__(self, f):
        """
        Apply the decorator to a function.

        Parameters
        ----------
        f : callable
            The function to decorate.

        Returns
        -------
        callable
            The original function if the CUBLAS version requirement is met,
            otherwise a placeholder function that raises `NotImplementedError`.
        """
        def f_new(*args, **kwargs):
            raise NotImplementedError("CUBLAS " + self.vs + " required")

        f_new.__doc__ = f.__doc__

        if _cublas_version >= int(self.vi):
            return f
        else:
            return f_new


_libcublas.cublasSetStream_v2.restype = int
_libcublas.cublasSetStream_v2.argtypes = [_types.handle, _types.stream]


def cublasSetStream(handle, id):
    """
    Set the current CUBLAS library stream.

    Associates the CUBLAS context with a specific CUDA stream for
    operation execution.

    Parameters
    ----------
    handle : int
        The CUBLAS context handle.
    id : int
        The stream ID to set.

    References
    ----------
    `cublasSetStream <http://docs.nvidia.com/cuda/cublas/#cublassetstream>`_
    """

    assert _libcublas
    status = _libcublas.cublasSetStream_v2(handle, id)
    cublasCheckStatus(status)


_libcublas.cublasGetStream_v2.restype = int
_libcublas.cublasGetStream_v2.argtypes = [_types.handle, ctypes.c_void_p]


def cublasGetStream(handle):
    """
    Get the current CUBLAS library stream.

    Retrieves the stream ID currently associated with the CUBLAS context.

    Parameters
    ----------
    handle : int
        The CUBLAS context handle.

    Returns
    -------
    id : int
        The current stream ID.

    References
    ----------
    `cublasGetStream <http://docs.nvidia.com/cuda/cublas/#cublasgetstream>`_
    """

    id = _types.stream()
    assert _libcublas
    status = _libcublas.cublasGetStream_v2(handle, ctypes.byref(id))
    cublasCheckStatus(status)
    return id.value


_libcublas.cublasGetPointerMode_v2.restype = int
_libcublas.cublasGetPointerMode_v2.argtypes = [_types.handle, ctypes.c_void_p]


def cublasGetPointerMode(handle):
    """
    Get the current CUBLAS pointer mode.

    Retrieves the pointer mode setting for the CUBLAS context. This
    determines whether scalar arguments are passed by value or by
    pointer.

    Parameters
    ----------
    handle : int
        The CUBLAS context handle.

    Returns
    -------
    mode : int
        The current pointer mode (e.g., `CUBLAS_POINTER_MODE_HOST` or
        `CUBLAS_POINTER_MODE_DEVICE`).

    """

    mode = ctypes.c_int()
    assert _libcublas
    status = _libcublas.cublasGetPointerMode_v2(handle, ctypes.byref(mode))
    cublasCheckStatus(status)
    return mode.value


_libcublas.cublasSetPointerMode_v2.restype = int
_libcublas.cublasSetPointerMode_v2.argtypes = [_types.handle, ctypes.c_int]


def cublasSetPointerMode(handle, mode):
    """
    Set the CUBLAS pointer mode.

    Sets the pointer mode for the CUBLAS context. This determines whether
    scalar arguments are passed by value or by pointer.

    Parameters
    ----------
    handle : int
        The CUBLAS context handle.
    mode : int
        The pointer mode to set (e.g., `CUBLAS_POINTER_MODE_HOST` or
        `CUBLAS_POINTER_MODE_DEVICE`).

    """

    assert _libcublas
    status = _libcublas.cublasSetPointerMode_v2(handle, mode)
    cublasCheckStatus(status)


# BLAS Level 1 Functions


# ISAMAX, IDAMAX, ICAMAX, IZAMAX
I_AMAX_doc = Template(
    """
    Finds the index of the maximum magnitude element.

    Finds the smallest index of the maximum magnitude element of a
    ${precision} ${real} vector.

    Note: for complex arguments x, the "magnitude" is defined as
    `abs(x.real) + abs(x.imag)`, *not* as `abs(x)`.

    Parameters
    ----------
    handle : int
        CUBLAS context.
    n : int
        Number of elements in input vector.
    x : ctypes.c_void_p
        Pointer to ${precision} ${real} input vector.
    incx : int
        Storage spacing between elements of `x`.

    Returns
    -------
    idx : int
        Index of maximum magnitude element (0-based).

    Examples
    --------
    >>> import pycuda.autoinit
    >>> import pycuda.gpuarray as gpuarray
    >>> import numpy as np
    >>> x = ${data}
    >>> x_gpu = gpuarray.to_gpu(x)
    >>> h = cublasCreate()
    >>> m = ${func}(h, x_gpu.size, x_gpu.gpudata, 1)
    >>> cublasDestroy(h)
    >>> np.allclose(m, np.argmax(abs(x.real) + abs(x.imag)))
    True

    Notes
    -----
    This function returns a 0-based index.

    References
    ----------
    `cublasI<t>amax <http://docs.nvidia.com/cuda/cublas/#cublasi-lt-t-gt-amax>`_
"""
)

_libcublas.cublasIsamax_v2.restype = int
_libcublas.cublasIsamax_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]


def cublasIsamax(handle, n, x, incx):
    """Finds the index of the maximum magnitude element (single precision real)."""
    result = ctypes.c_int()
    assert _libcublas
    status = _libcublas.cublasIsamax_v2(handle, n, int(x), incx, ctypes.byref(result))
    cublasCheckStatus(status)
    return result.value - 1


cublasIsamax.__doc__ = I_AMAX_doc.substitute(precision="single precision", real="real", data="np.random.rand(5).astype(np.float32)", func="cublasIsamax")

_libcublas.cublasIdamax_v2.restype = int
_libcublas.cublasIdamax_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]


def cublasIdamax(handle, n, x, incx):
    """Finds the index of the maximum magnitude element (double precision real)."""
    result = ctypes.c_int()
    assert _libcublas
    status = _libcublas.cublasIdamax_v2(handle, n, int(x), incx, ctypes.byref(result))
    cublasCheckStatus(status)
    return result.value - 1


cublasIdamax.__doc__ = I_AMAX_doc.substitute(precision="double precision", real="real", data="np.random.rand(5).astype(np.float64)", func="cublasIdamax")

_libcublas.cublasIcamax_v2.restype = int
_libcublas.cublasIcamax_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]


def cublasIcamax(handle, n, x, incx):
    """Finds the index of the maximum magnitude element (single precision complex)."""
    result = ctypes.c_int()
    assert _libcublas
    status = _libcublas.cublasIcamax_v2(handle, n, int(x), incx, ctypes.byref(result))
    cublasCheckStatus(status)
    return result.value - 1


cublasIcamax.__doc__ = I_AMAX_doc.substitute(precision="single precision", real="complex", data="(np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex64)", func="cublasIcamax")

_libcublas.cublasIzamax_v2.restype = int
_libcublas.cublasIzamax_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]


def cublasIzamax(handle, n, x, incx):
    """Finds the index of the maximum magnitude element (double precision complex)."""
    result = ctypes.c_int()
    assert _libcublas
    status = _libcublas.cublasIzamax_v2(handle, n, int(x), incx, ctypes.byref(result))
    cublasCheckStatus(status)
    return result.value - 1


cublasIzamax.__doc__ = I_AMAX_doc.substitute(precision="double precision", real="complex", data="(np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex128)", func="cublasIzamax")

# ISAMIN, IDAMIN, ICAMIN, IZAMIN
I_AMIN_doc = Template(
    """
    Finds the index of the minimum magnitude element.

    Finds the smallest index of the minimum magnitude element of a
    ${precision} ${real} vector.

    Note: for complex arguments x, the "magnitude" is defined as
    `abs(x.real) + abs(x.imag)`, *not* as `abs(x)`.

    Parameters
    ----------
    handle : int
        CUBLAS context.
    n : int
        Number of elements in input vector.
    x : ctypes.c_void_p
        Pointer to ${precision} ${real} input vector.
    incx : int
        Storage spacing between elements of `x`.

    Returns
    -------
    idx : int
        Index of minimum magnitude element (0-based).

    Examples
    --------
    >>> import pycuda.autoinit
    >>> import pycuda.gpuarray as gpuarray
    >>> import numpy as np
    >>> x = ${data}
    >>> x_gpu = gpuarray.to_gpu(x)
    >>> h = cublasCreate()
    >>> m = ${func}(h, x_gpu.size, x_gpu.gpudata, 1)
    >>> cublasDestroy(h)
    >>> np.allclose(m, np.argmin(abs(x.real) + abs(x.imag)))
    True

    Notes
    -----
    This function returns a 0-based index.

    References
    ----------
    `cublasI<t>amin <http://docs.nvidia.com/cuda/cublas/#cublasi-lt-t-gt-amin>`_
    """
)

_libcublas.cublasIsamin_v2.restype = int
_libcublas.cublasIsamin_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]


def cublasIsamin(handle, n, x, incx):
    """Finds the index of the minimum magnitude element (single precision real)."""
    result = ctypes.c_int()
    assert _libcublas
    status = _libcublas.cublasIsamin_v2(handle, n, int(x), incx, ctypes.byref(result))
    cublasCheckStatus(status)
    return result.value - 1


cublasIsamin.__doc__ = I_AMIN_doc.substitute(precision="single precision", real="real", data="np.random.rand(5).astype(np.float32)", func="cublasIsamin")

_libcublas.cublasIdamin_v2.restype = int
_libcublas.cublasIdamin_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]


def cublasIdamin(handle, n, x, incx):
    """Finds the index of the minimum magnitude element (double precision real)."""
    result = ctypes.c_int()
    assert _libcublas
    status = _libcublas.cublasIdamin_v2(handle, n, int(x), incx, ctypes.byref(result))
    cublasCheckStatus(status)
    return result.value - 1


cublasIdamin.__doc__ = I_AMIN_doc.substitute(precision="double precision", real="real", data="np.random.rand(5).astype(np.float64)", func="cublasIdamin")

_libcublas.cublasIcamin_v2.restype = int
_libcublas.cublasIcamin_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]


def cublasIcamin(handle, n, x, incx):
    """Finds the index of the minimum magnitude element (single precision complex)."""
    result = ctypes.c_int()
    assert _libcublas
    status = _libcublas.cublasIcamin_v2(handle, n, int(x), incx, ctypes.byref(result))
    cublasCheckStatus(status)
    return result.value - 1


cublasIcamin.__doc__ = I_AMIN_doc.substitute(precision="single precision", real="complex", data="(np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex64)", func="cublasIcamin")

_libcublas.cublasIzamin_v2.restype = int
_libcublas.cublasIzamin_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]


def cublasIzamin(handle, n, x, incx):
    """Finds the index of the minimum magnitude element (double precision complex)."""
    result = ctypes.c_int()
    assert _libcublas
    status = _libcublas.cublasIzamin_v2(handle, n, int(x), incx, ctypes.byref(result))
    cublasCheckStatus(status)
    return result.value - 1


cublasIzamin.__doc__ = I_AMIN_doc.substitute(precision="double precision", real="complex", data="(np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex128)", func="cublasIzamin")

# SASUM, DASUM, SCASUM, DZASUM
_ASUM_doc = Template(
    """
    Computes the sum of the absolute values of vector elements.

    Computes the sum of the absolute values of the elements of a
    ${precision} ${real} vector.

    Note: if the vector is complex, then this computes the sum
    `sum(abs(x.real)) + sum(abs(x.imag))`

    Parameters
    ----------
    handle : int
        CUBLAS context.
    n : int
        Number of elements in input vector.
    x : ctypes.c_void_p
        Pointer to ${precision} ${real} input vector.
    incx : int
        Storage spacing between elements of `x`.

    Examples
    --------
    >>> import pycuda.autoinit
    >>> import pycuda.gpuarray as gpuarray
    >>> import numpy as np
    >>> x = ${data}
    >>> x_gpu = gpuarray.to_gpu(x)
    >>> h = cublasCreate()
    >>> s = ${func}(h, x_gpu.size, x_gpu.gpudata, 1)
    >>> cublasDestroy(h)
    >>> np.allclose(s, abs(x.real).sum() + abs(x.imag).sum())
    True

    Returns
    -------
    s : ${ret_type}
        Sum of absolute values.

    References
    ----------
    `cublas<t>sum <http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-asum>`_
    """
)

_libcublas.cublasSasum_v2.restype = int
_libcublas.cublasSasum_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]


def cublasSasum(handle, n, x, incx):
    """Computes the sum of absolute values (single precision real)."""
    result = ctypes.c_float()
    assert _libcublas
    status = _libcublas.cublasSasum_v2(handle, n, int(x), incx, ctypes.byref(result))
    cublasCheckStatus(status)
    return np.float32(result.value)


cublasSasum.__doc__ = _ASUM_doc.substitute(precision="single precision", real="real", data="np.random.rand(5).astype(np.float32)", func="cublasSasum", ret_type="numpy.float32")

_libcublas.cublasDasum_v2.restype = int
_libcublas.cublasDasum_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]


def cublasDasum(handle, n, x, incx):
    """Computes the sum of absolute values (double precision real)."""
    result = ctypes.c_double()
    assert _libcublas
    status = _libcublas.cublasDasum_v2(handle, n, int(x), incx, ctypes.byref(result))
    cublasCheckStatus(status)
    return np.float64(result.value)


cublasDasum.__doc__ = _ASUM_doc.substitute(precision="double precision", real="real", data="np.random.rand(5).astype(np.float64)", func="cublasDasum", ret_type="numpy.float64")

_libcublas.cublasScasum_v2.restype = int
_libcublas.cublasScasum_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]


def cublasScasum(handle, n, x, incx):
    """Computes the sum of absolute values (single precision complex)."""
    result = ctypes.c_float()
    assert _libcublas
    status = _libcublas.cublasScasum_v2(handle, n, int(x), incx, ctypes.byref(result))
    cublasCheckStatus(status)
    return np.float32(result.value)


cublasScasum.__doc__ = _ASUM_doc.substitute(
    precision="single precision", real="complex", data="(np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex64)", func="cublasScasum", ret_type="numpy.float32"
)

_libcublas.cublasDzasum_v2.restype = int
_libcublas.cublasDzasum_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]


def cublasDzasum(handle, n, x, incx):
    """Computes the sum of absolute values (double precision complex)."""
    result = ctypes.c_double()
    assert _libcublas
    status = _libcublas.cublasDzasum_v2(handle, n, int(x), incx, ctypes.byref(result))
    cublasCheckStatus(status)
    return np.float64(result.value)


cublasDzasum.__doc__ = _ASUM_doc.substitute(
    precision="double precision", real="complex", data="(np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex128)", func="cublasDzasum", ret_type="numpy.float64"
)

# SAXPY, DAXPY, CAXPY, ZAXPY
_AXPY_doc = Template(
    """
    Computes the sum of a scaled vector and another vector.

    Computes the sum of a ${precision} ${real} vector scaled by a
    ${precision} ${real} scalar and another ${precision} ${real} vector.
    The result is stored in the second vector.

    Parameters
    ----------
    handle : int
        CUBLAS context.
    n : int
        Number of elements in input vectors.
    alpha : ${type}
        Scalar multiplier for vector `x`.
    x : ctypes.c_void_p
        Pointer to the first input vector.
    incx : int
        Storage spacing between elements of `x`.
    y : ctypes.c_void_p
        Pointer to the second input/output vector. The result is stored here.
    incy : int
        Storage spacing between elements of `y`.

    Examples
    --------
    >>> import pycuda.autoinit
    >>> import pycuda.gpuarray as gpuarray
    >>> import numpy as np
    >>> alpha = ${alpha}
    >>> x = ${data}
    >>> y = ${data}
    >>> x_gpu = gpuarray.to_gpu(x)
    >>> y_gpu = gpuarray.to_gpu(y)
    >>> h = cublasCreate()
    >>> ${func}(h, x_gpu.size, alpha, x_gpu.gpudata, 1, y_gpu.gpudata, 1)
    >>> cublasDestroy(h)
    >>> np.allclose(y_gpu.get(), alpha*x+y)
    True

    Notes
    -----
    Both `x` and `y` must contain `n` elements.

    References
    ----------
    `cublas<t>axpy <http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-axpy>`_
    """
)

_libcublas.cublasSaxpy_v2.restype = int
_libcublas.cublasSaxpy_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]


def cublasSaxpy(handle, n, alpha, x, incx, y, incy):
    """Computes y = alpha*x + y (single precision real)."""
    assert _libcublas
    status = _libcublas.cublasSaxpy_v2(handle, n, ctypes.byref(ctypes.c_float(alpha)), int(x), incx, int(y), incy)
    cublasCheckStatus(status)


cublasSaxpy.__doc__ = _AXPY_doc.substitute(
    precision="single precision", real="real", type="numpy.float32", alpha="np.float32(np.random.rand())", data="np.random.rand(5).astype(np.float32)", func="cublasSaxpy"
)

_libcublas.cublasDaxpy_v2.restype = int
_libcublas.cublasDaxpy_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]


def cublasDaxpy(handle, n, alpha, x, incx, y, incy):
    """Computes y = alpha*x + y (double precision real)."""
    assert _libcublas
    status = _libcublas.cublasDaxpy_v2(handle, n, ctypes.byref(ctypes.c_double(alpha)), int(x), incx, int(y), incy)
    cublasCheckStatus(status)


cublasDaxpy.__doc__ = _AXPY_doc.substitute(
    precision="double precision", real="real", type="numpy.float64", alpha="np.float64(np.random.rand())", data="np.random.rand(5).astype(np.float64)", func="cublasDaxpy"
)

_libcublas.cublasCaxpy_v2.restype = int
_libcublas.cublasCaxpy_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]


def cublasCaxpy(handle, n, alpha, x, incx, y, incy):
    """Computes y = alpha*x + y (single precision complex)."""
    assert _libcublas
    status = _libcublas.cublasCaxpy_v2(handle, n, ctypes.byref(cuda.cuFloatComplex(alpha.real, alpha.imag)), int(x), incx, int(y), incy)
    cublasCheckStatus(status)


cublasCaxpy.__doc__ = _AXPY_doc.substitute(
    precision="single precision",
    real="complex",
    type="numpy.complex64",
    alpha="np.complex64(np.random.rand()+1j*np.random.rand())",
    data="(np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex64)",
    func="cublasCaxpy",
)

_libcublas.cublasZaxpy_v2.restype = int
_libcublas.cublasZaxpy_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]


def cublasZaxpy(handle, n, alpha, x, incx, y, incy):
    """Computes y = alpha*x + y (double precision complex)."""
    assert _libcublas
    status = _libcublas.cublasZaxpy_v2(handle, n, ctypes.byref(cuda.cuDoubleComplex(alpha.real, alpha.imag)), int(x), incx, int(y), incy)
    cublasCheckStatus(status)


cublasZaxpy.__doc__ = _AXPY_doc.substitute(
    precision="double precision",
    real="complex",
    type="numpy.complex128",
    alpha="np.complex128(np.random.rand()+1j*np.random.rand())",
    data="(np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex128)",
    func="cublasZaxpy",
)

# SCOPY, DCOPY, CCOPY, ZCOPY
_COPY_doc = Template(
    """
    Copies a vector.

    Copies a ${precision} ${real} vector to another ${precision} ${real}
    vector.

    Parameters
    ----------
    handle : int
        CUBLAS context.
    n : int
        Number of elements in input vectors.
    x : ctypes.c_void_p
        Pointer to the input vector.
    incx : int
        Storage spacing between elements of `x`.
    y : ctypes.c_void_p
        Pointer to the output vector.
    incy : int
        Storage spacing between elements of `y`.

    Examples
    --------
    >>> import pycuda.autoinit
    >>> import pycuda.gpuarray as gpuarray
    >>> import numpy as np
    >>> x = ${data}
    >>> x_gpu = gpuarray.to_gpu(x)
    >>> y_gpu = gpuarray.zeros_like(x_gpu)
    >>> h = cublasCreate()
    >>> ${func}(h, x_gpu.size, x_gpu.gpudata, 1, y_gpu.gpudata, 1)
    >>> cublasDestroy(h)
    >>> np.allclose(y_gpu.get(), x_gpu.get())
    True

    Notes
    -----
    Both `x` and `y` must contain `n` elements.

    References
    ----------
    `cublas<t>copy <http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-copy>`_
"""
)

_libcublas.cublasScopy_v2.restype = int
_libcublas.cublasScopy_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]


def cublasScopy(handle, n, x, incx, y, incy):
    """Copies vector x to vector y (single precision real)."""
    assert _libcublas
    status = _libcublas.cublasScopy_v2(handle, n, int(x), incx, int(y), incy)
    cublasCheckStatus(status)


cublasScopy.__doc__ = _COPY_doc.substitute(precision="single precision", real="real", data="np.random.rand(5).astype(np.float32)", func="cublasScopy")

_libcublas.cublasDcopy_v2.restype = int
_libcublas.cublasDcopy_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]


def cublasDcopy(handle, n, x, incx, y, incy):
    """Copies vector x to vector y (double precision real)."""
    assert _libcublas
    status = _libcublas.cublasDcopy_v2(handle, n, int(x), incx, int(y), incy)
    cublasCheckStatus(status)


cublasDcopy.__doc__ = _COPY_doc.substitute(precision="double precision", real="real", data="np.random.rand(5).astype(np.float64)", func="cublasDcopy")

_libcublas.cublasCcopy_v2.restype = int
_libcublas.cublasCcopy_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]


def cublasCcopy(handle, n, x, incx, y, incy):
    """Copies vector x to vector y (single precision complex)."""
    assert _libcublas
    status = _libcublas.cublasCcopy_v2(handle, n, int(x), incx, int(y), incy)
    cublasCheckStatus(status)


cublasCcopy.__doc__ = _COPY_doc.substitute(precision="single precision", real="complex", data="(np.random.rand(5)+np.random.rand(5)).astype(np.complex64)", func="cublasCcopy")

_libcublas.cublasZcopy_v2.restype = int
_libcublas.cublasZcopy_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]


def cublasZcopy(handle, n, x, incx, y, incy):
    """Copies vector x to vector y (double precision complex)."""
    assert _libcublas
    status = _libcublas.cublasZcopy_v2(handle, n, int(x), incx, int(y), incy)
    cublasCheckStatus(status)


cublasZcopy.__doc__ = _COPY_doc.substitute(precision="double precision", real="complex", data="(np.random.rand(5)+np.random.rand(5)).astype(np.complex128)", func="cublasZcopy")

# SDOT, DDOT, CDOTU, CDOTC, ZDOTU, ZDOTC
_DOT_doc = Template(
    """
    Computes the dot product of two vectors.

    Computes the dot product of two ${precision} ${real} vectors.
    `cublasCdotc` and `cublasZdotc` use the conjugate of the first vector
    when computing the dot product.

    Parameters
    ----------
    handle : int
        CUBLAS context.
    n : int
        Number of elements in input vectors.
    x : ctypes.c_void_p
        Pointer to the first input vector.
    incx : int
        Storage spacing between elements of `x`.
    y : ctypes.c_void_p
        Pointer to the second input vector.
    incy : int
        Storage spacing between elements of `y`.

    Returns
    -------
    d : ${ret_type}
        Dot product of `x` and `y`.

    Examples
    --------
    >>> import pycuda.autoinit
    >>> import pycuda.gpuarray as gpuarray
    >>> import numpy as np
    >>> x = ${data}
    >>> y = ${data}
    >>> x_gpu = gpuarray.to_gpu(x)
    >>> y_gpu = gpuarray.to_gpu(y)
    >>> h = cublasCreate()
    >>> d = ${func}(h, x_gpu.size, x_gpu.gpudata, 1, y_gpu.gpudata, 1)
    >>> cublasDestroy(h)
    >>> ${check}
    True

    Notes
    -----
    Both `x` and `y` must contain `n` elements.

    References
    ----------
    `cublas<t>dot <http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-dot>`_
"""
)

_libcublas.cublasSdot_v2.restype = int
_libcublas.cublasSdot_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]


def cublasSdot(handle, n, x, incx, y, incy):
    """Computes the dot product of two vectors (single precision real)."""
    result = ctypes.c_float()
    assert _libcublas
    status = _libcublas.cublasSdot_v2(handle, n, int(x), incx, int(y), incy, ctypes.byref(result))
    cublasCheckStatus(status)
    return np.float32(result.value)


cublasSdot.__doc__ = _DOT_doc.substitute(
    precision="single precision", real="real", data="np.float32(np.random.rand(5))", ret_type="np.float32", func="cublasSdot", check="np.allclose(d, np.dot(x, y))"
)

_libcublas.cublasDdot_v2.restype = int
_libcublas.cublasDdot_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]


def cublasDdot(handle, n, x, incx, y, incy):
    """Computes the dot product of two vectors (double precision real)."""
    result = ctypes.c_double()
    assert _libcublas
    status = _libcublas.cublasDdot_v2(handle, n, int(x), incx, int(y), incy, ctypes.byref(result))
    cublasCheckStatus(status)
    return np.float64(result.value)


cublasDdot.__doc__ = _DOT_doc.substitute(
    precision="double precision", real="real", data="np.float64(np.random.rand(5))", ret_type="np.float64", func="cublasDdot", check="np.allclose(d, np.dot(x, y))"
)

_libcublas.cublasCdotu_v2.restype = int
_libcublas.cublasCdotu_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]


def cublasCdotu(handle, n, x, incx, y, incy):
    """Computes the dot product of two vectors (single precision complex, non-conjugate)."""
    result = cuda.cuFloatComplex()
    assert _libcublas
    status = _libcublas.cublasCdotu_v2(handle, n, int(x), incx, int(y), incy, ctypes.byref(result))
    cublasCheckStatus(status)
    return np.complex64(result.value)


cublasCdotu.__doc__ = _DOT_doc.substitute(
    precision="single precision",
    real="complex",
    data="(np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex64)",
    ret_type="np.complex64",
    func="cublasCdotu",
    check="np.allclose(d, np.dot(x, y))",
)

_libcublas.cublasCdotc_v2.restype = int
_libcublas.cublasCdotc_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]


def cublasCdotc(handle, n, x, incx, y, incy):
    """Computes the dot product of two vectors (single precision complex, conjugate)."""
    result = cuda.cuFloatComplex()
    assert _libcublas
    status = _libcublas.cublasCdotc_v2(handle, n, int(x), incx, int(y), incy, ctypes.byref(result))
    cublasCheckStatus(status)
    return np.complex64(result.value)


cublasCdotc.__doc__ = _DOT_doc.substitute(
    precision="single precision",
    real="complex",
    data="(np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex64)",
    ret_type="np.complex64",
    func="cublasCdotc",
    check="np.allclose(d, np.dot(np.conj(x), y))",
)

_libcublas.cublasZdotu_v2.restype = int
_libcublas.cublasZdotu_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]


def cublasZdotu(handle, n, x, incx, y, incy):
    """Computes the dot product of two vectors (double precision complex, non-conjugate)."""
    result = cuda.cuDoubleComplex()
    assert _libcublas
    status = _libcublas.cublasZdotu_v2(handle, n, int(x), incx, int(y), incy, ctypes.byref(result))
    cublasCheckStatus(status)
    return np.complex128(result.value)


cublasZdotu.__doc__ = _DOT_doc.substitute(
    precision="double precision",
    real="complex",
    data="(np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex128)",
    ret_type="np.complex128",
    func="cublasZdotu",
    check="np.allclose(d, np.dot(x, y))",
)

_libcublas.cublasZdotc_v2.restype = int
_libcublas.cublasZdotc_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]


def cublasZdotc(handle, n, x, incx, y, incy):
    """Computes the dot product of two vectors (double precision complex, conjugate)."""
    result = cuda.cuDoubleComplex()
    assert _libcublas
    status = _libcublas.cublasZdotc_v2(handle, n, int(x), incx, int(y), incy, ctypes.byref(result))
    cublasCheckStatus(status)
    return np.complex128(result.value)


cublasZdotc.__doc__ = _DOT_doc.substitute(
    precision="double precision",
    real="complex",
    data="(np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex128)",
    ret_type="np.complex128",
    func="cublasZdotc",
    check="np.allclose(d, np.dot(np.conj(x), y))",
)

# SNRM2, DNRM2, SCNRM2, DZNRM2
_NRM2_doc = Template(
    """
    Computes the Euclidean norm (2-norm) of a vector.

    Computes the Euclidean norm of a ${precision} ${real} vector.

    Parameters
    ----------
    handle : int
        CUBLAS context.
    n : int
        Number of elements in input vector.
    x : ctypes.c_void_p
        Pointer to the input vector.
    incx : int
        Storage spacing between elements of `x`.

    Returns
    -------
    nrm : ${ret_type}
        Euclidean norm of `x`.

    Examples
    --------
    >>> import pycuda.autoinit
    >>> import pycuda.gpuarray as gpuarray
    >>> import numpy as np
    >>> x = ${data}
    >>> x_gpu = gpuarray.to_gpu(x)
    >>> h = cublasCreate()
    >>> nrm = ${func}(h, x.size, x_gpu.gpudata, 1)
    >>> cublasDestroy(h)
    >>> np.allclose(nrm, np.linalg.norm(x))
    True

    References
    ----------
    `cublas<t>nrm2 <http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-nrm2>`_
"""
)

_libcublas.cublasSnrm2_v2.restype = int
_libcublas.cublasSnrm2_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]


def cublasSnrm2(handle, n, x, incx):
    """Computes the Euclidean norm of a vector (single precision real)."""
    result = ctypes.c_float()
    assert _libcublas
    status = _libcublas.cublasSnrm2_v2(handle, n, int(x), incx, ctypes.byref(result))
    cublasCheckStatus(status)
    return np.float32(result.value)


cublasSnrm2.__doc__ = _NRM2_doc.substitute(precision="single precision", real="real", data="np.float32(np.random.rand(5))", ret_type="numpy.float32", func="cublasSnrm2")

_libcublas.cublasDnrm2_v2.restype = int
_libcublas.cublasDnrm2_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]


def cublasDnrm2(handle, n, x, incx):
    """Computes the Euclidean norm of a vector (double precision real)."""
    result = ctypes.c_double()
    assert _libcublas
    status = _libcublas.cublasDnrm2_v2(handle, n, int(x), incx, ctypes.byref(result))
    cublasCheckStatus(status)
    return np.float64(result.value)


cublasDnrm2.__doc__ = _NRM2_doc.substitute(precision="double precision", real="real", data="np.float64(np.random.rand(5))", ret_type="numpy.float64", func="cublasDnrm2")

_libcublas.cublasScnrm2_v2.restype = int
_libcublas.cublasScnrm2_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]


def cublasScnrm2(handle, n, x, incx):
    """Computes the Euclidean norm of a vector (single precision complex)."""
    result = ctypes.c_float()
    assert _libcublas
    status = _libcublas.cublasScnrm2_v2(handle, n, int(x), incx, ctypes.byref(result))
    cublasCheckStatus(status)
    return np.float32(result.value)


cublasScnrm2.__doc__ = _NRM2_doc.substitute(
    precision="single precision", real="complex", data="(np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex64)", ret_type="numpy.complex64", func="cublasScnrm2"
)

_libcublas.cublasDznrm2_v2.restype = int
_libcublas.cublasDznrm2_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]


def cublasDznrm2(handle, n, x, incx):
    """Computes the Euclidean norm of a vector (double precision complex)."""
    result = ctypes.c_double()
    assert _libcublas
    status = _libcublas.cublasDznrm2_v2(handle, n, int(x), incx, ctypes.byref(result))
    cublasCheckStatus(status)
    return np.float64(result.value)


cublasDznrm2.__doc__ = _NRM2_doc.substitute(
    precision="double precision", real="complex", data="(np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex128)", ret_type="numpy.complex128", func="cublasDznrm2"
)


# SROT, DROT, CROT, CSROT, ZROT, ZDROT
_ROT_doc = Template(
    """
    Applies a Givens rotation to two vectors.

    Multiplies the ${precision} matrix `[[c, s], [-s.conj(), c]]`
    with the 2 x `n` ${precision} matrix `[[x.T], [y.T]]`.

    Parameters
    ----------
    handle : int
        CUBLAS context.
    n : int
        Number of elements in input vectors.
    x : ctypes.c_void_p
        Pointer to the first input/output vector.
    incx : int
        Storage spacing between elements of `x`.
    y : ctypes.c_void_p
        Pointer to the second input/output vector.
    incy : int
        Storage spacing between elements of `y`.
    c : ${c_type}
        Cosine component of the rotation.
    s : ${s_type}
        Sine component of the rotation.

    Notes
    -----
    Both `x` and `y` must contain `n` elements.

    Examples
    --------
    >>> import pycuda.autoinit
    >>> import pycuda.gpuarray as gpuarray
    >>> import numpy as np
    >>> s = ${s_val}; c = ${c_val};
    >>> x = ${data}
    >>> y = ${data}
    >>> x_gpu = gpuarray.to_gpu(x)
    >>> y_gpu = gpuarray.to_gpu(y)
    >>> h = cublasCreate()
    >>> ${func}(h, x.size, x_gpu.gpudata, 1, y_gpu.gpudata, 1, c, s)
    >>> cublasDestroy(h)
    >>> np.allclose(x_gpu.get(), c*x+s*y)
    True
    >>> np.allclose(y_gpu.get(), -s.conj()*x+c*y)
    True

    References
    ----------
    `cublas<t>rot <http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-rot>`_
"""
)

_libcublas.cublasSrot_v2.restype = int
_libcublas.cublasSrot_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]


def cublasSrot(handle, n, x, incx, y, incy, c, s):
    """Applies a Givens rotation to two real vectors (single precision)."""
    assert _libcublas
    status = _libcublas.cublasSrot_v2(handle, n, int(x), incx, int(y), incy, ctypes.byref(ctypes.c_float(c)), ctypes.byref(ctypes.c_float(s)))

    cublasCheckStatus(status)


cublasSrot.__doc__ = _ROT_doc.substitute(
    precision="single precision",
    real="real",
    c_type="numpy.float32",
    s_type="numpy.float32",
    c_val="np.float32(np.random.rand())",
    s_val="np.float32(np.random.rand())",
    data="np.random.rand(5).astype(np.float32)",
    func="cublasSrot",
)

_libcublas.cublasDrot_v2.restype = int
_libcublas.cublasDrot_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]


def cublasDrot(handle, n, x, incx, y, incy, c, s):
    """Applies a Givens rotation to two real vectors (double precision)."""
    assert _libcublas
    status = _libcublas.cublasDrot_v2(handle, n, int(x), incx, int(y), incy, ctypes.byref(ctypes.c_double(c)), ctypes.byref(ctypes.c_double(s)))
    cublasCheckStatus(status)


cublasDrot.__doc__ = _ROT_doc.substitute(
    precision="double precision",
    real="real",
    c_type="numpy.float64",
    s_type="numpy.float64",
    c_val="np.float64(np.random.rand())",
    s_val="np.float64(np.random.rand())",
    data="np.random.rand(5).astype(np.float64)",
    func="cublasDrot",
)

_libcublas.cublasCrot_v2.restype = int
_libcublas.cublasCrot_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]


def cublasCrot(handle, n, x, incx, y, incy, c, s):
    """Applies a Givens rotation to two complex vectors (single precision)."""
    assert _libcublas
    status = _libcublas.cublasCrot_v2(handle, n, int(x), incx, int(y), incy, ctypes.byref(ctypes.c_float(c)), ctypes.byref(cuda.cuFloatComplex(s.real, s.imag)))
    cublasCheckStatus(status)


cublasCrot.__doc__ = _ROT_doc.substitute(
    precision="single precision",
    real="complex",
    c_type="numpy.float32",
    s_type="numpy.complex64",
    c_val="np.float32(np.random.rand())",
    s_val="np.complex64(np.random.rand()+1j*np.random.rand())",
    data="(np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex64)",
    func="cublasCrot",
)

_libcublas.cublasCsrot_v2.restype = int
_libcublas.cublasCsrot_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]


def cublasCsrot(handle, n, x, incx, y, incy, c, s):
    """Applies a Givens rotation to two complex vectors (single precision real scalar)."""
    assert _libcublas
    status = _libcublas.cublasCsrot_v2(handle, n, int(x), incx, int(y), incy, ctypes.byref(ctypes.c_float(c)), ctypes.byref(ctypes.c_float(s)))
    cublasCheckStatus(status)


cublasCsrot.__doc__ = _ROT_doc.substitute(
    precision="single precision",
    real="complex",
    c_type="numpy.float32",
    s_type="numpy.float32",
    c_val="np.float32(np.random.rand())",
    s_val="np.float32(np.random.rand())",
    data="(np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex64)",
    func="cublasCsrot",
)

_libcublas.cublasZrot_v2.restype = int
_libcublas.cublasZrot_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]


def cublasZrot(handle, n, x, incx, y, incy, c, s):
    """Applies a Givens rotation to two complex vectors (double precision)."""
    assert _libcublas
    status = _libcublas.cublasZrot_v2(handle, n, int(x), incx, int(y), incy, ctypes.byref(ctypes.c_double(c)), ctypes.byref(cuda.cuDoubleComplex(s.real, s.imag)))
    cublasCheckStatus(status)


cublasZrot.__doc__ = _ROT_doc.substitute(
    precision="double precision",
    real="complex",
    c_type="numpy.float64",
    s_type="numpy.complex128",
    c_val="np.float64(np.random.rand())",
    s_val="np.complex128(np.random.rand()+1j*np.random.rand())",
    data="(np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex128)",
    func="cublasZrot",
)

_libcublas.cublasZdrot_v2.restype = int
_libcublas.cublasZdrot_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]


def cublasZdrot(handle, n, x, incx, y, incy, c, s):
    """Applies a Givens rotation to two complex vectors (double precision real scalar)."""
    assert _libcublas
    status = _libcublas.cublasZdrot_v2(handle, n, int(x), incx, int(y), incy, ctypes.byref(ctypes.c_double(c)), ctypes.byref(ctypes.c_double(s)))
    cublasCheckStatus(status)


cublasZdrot.__doc__ = _ROT_doc.substitute(
    precision="double precision",
    real="complex",
    c_type="numpy.float64",
    s_type="numpy.float64",
    c_val="np.float64(np.random.rand())",
    s_val="np.float64(np.random.rand())",
    data="(np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex128)",
    func="cublasZdrot",
)


# SROTG, DROTG, CROTG, ZROTG
_ROTG_doc = Template(
    """
    Constructs a Givens rotation matrix.

    Constructs the ${precision} ${real} Givens rotation matrix
    `G = [[c, s], [-s.conj(), c]]` such that
    `dot(G, [[a], [b]]) == [[r], [0]]`, where
    `c**2+s**2 == 1` and `r == a**2+b**2` for real numbers and
    `c**2+(conj(s)*s) == 1` and `r ==
    (a/abs(a))*sqrt(abs(a)**2+abs(b)**2)` for `a != 0` and `r == b`
    for `a == 0`.

    Parameters
    ----------
    handle : int
        CUBLAS context.
    a : ${type}
        First element of the input vector.
    b : ${type}
        Second element of the input vector.

    Returns
    -------
    r : ${type}
        The norm of the input vector `[[a], [b]]`.
    c : ${c_type}
        The cosine component of the rotation matrix.
    s : ${s_type}
        The sine component of the rotation matrix.

    Examples
    --------
    >>> import pycuda.autoinit
    >>> import pycuda.gpuarray as gpuarray
    >>> import numpy as np
    >>> a = ${a_val}
    >>> b = ${b_val}
    >>> h = cublasCreate()
    >>> r, c, s = ${func}(h, a, b)
    >>> cublasDestroy(h)
    >>> np.allclose(np.dot(np.array([[c, s], [-np.conj(s), c]]), np.array([[a], [b]])), np.array([[r], [0.0]]), atol=1e-6)
    True

    References
    ----------
    `cublas<t>rotg <http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-rotg>`_
"""
)

_libcublas.cublasSrotg_v2.restype = int
_libcublas.cublasSrotg_v2.argtypes = [_types.handle, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]


def cublasSrotg(handle, a, b):
    """Constructs a Givens rotation matrix (single precision real)."""
    _a = ctypes.c_float(a)
    _b = ctypes.c_float(b)
    _c = ctypes.c_float()
    _s = ctypes.c_float()
    assert _libcublas
    status = _libcublas.cublasSrotg_v2(handle, ctypes.byref(_a), ctypes.byref(_b), ctypes.byref(_c), ctypes.byref(_s))
    cublasCheckStatus(status)
    return np.float32(_a.value), np.float32(_c.value), np.float32(_s.value)


cublasSrotg.__doc__ = _ROTG_doc.substitute(
    precision="single precision",
    real="real",
    type="numpy.float32",
    c_type="numpy.float32",
    s_type="numpy.float32",
    a_val="np.float32(np.random.rand())",
    b_val="np.float32(np.random.rand())",
    func="cublasSrotg",
)

_libcublas.cublasDrotg_v2.restype = int
_libcublas.cublasDrotg_v2.argtypes = [_types.handle, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]


def cublasDrotg(handle, a, b):
    """Constructs a Givens rotation matrix (double precision real)."""
    _a = ctypes.c_double(a)
    _b = ctypes.c_double(b)
    _c = ctypes.c_double()
    _s = ctypes.c_double()
    assert _libcublas
    status = _libcublas.cublasDrotg_v2(handle, ctypes.byref(_a), ctypes.byref(_b), ctypes.byref(_c), ctypes.byref(_s))
    cublasCheckStatus(status)
    return np.float64(_a.value), np.float64(_c.value), np.float64(_s.value)


cublasDrotg.__doc__ = _ROTG_doc.substitute(
    precision="double precision",
    real="real",
    type="numpy.float64",
    c_type="numpy.float64",
    s_type="numpy.float64",
    a_val="np.float64(np.random.rand())",
    b_val="np.float64(np.random.rand())",
    func="cublasDrotg",
)

_libcublas.cublasCrotg_v2.restype = int
_libcublas.cublasCrotg_v2.argtypes = [_types.handle, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]


def cublasCrotg(handle, a, b):
    """Constructs a Givens rotation matrix (single precision complex)."""
    _a = cuda.cuFloatComplex(a.real, a.imag)
    _b = cuda.cuFloatComplex(b.real, b.imag)
    _c = ctypes.c_float()
    _s = cuda.cuFloatComplex()
    assert _libcublas
    status = _libcublas.cublasCrotg_v2(handle, ctypes.byref(_a), ctypes.byref(_b), ctypes.byref(_c), ctypes.byref(_s))
    cublasCheckStatus(status)
    return np.complex64(_a.value), np.float32(_c.value), np.complex64(_s.value)


cublasCrotg.__doc__ = _ROTG_doc.substitute(
    precision="single precision",
    real="complex",
    type="numpy.complex64",
    c_type="numpy.float32",
    s_type="numpy.complex64",
    a_val="np.complex64(np.random.rand()+1j*np.random.rand())",
    b_val="np.complex64(np.random.rand()+1j*np.random.rand())",
    func="cublasCrotg",
)

_libcublas.cublasZrotg_v2.restype = int
_libcublas.cublasZrotg_v2.argtypes = [_types.handle, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]


def cublasZrotg(handle, a, b):
    """Constructs a Givens rotation matrix (double precision complex)."""
    _a = cuda.cuDoubleComplex(a.real, a.imag)
    _b = cuda.cuDoubleComplex(b.real, b.imag)
    _c = ctypes.c_double()
    _s = cuda.cuDoubleComplex()
    assert _libcublas
    status = _libcublas.cublasZrotg_v2(handle, ctypes.byref(_a), ctypes.byref(_b), ctypes.byref(_c), ctypes.byref(_s))
    cublasCheckStatus(status)
    return np.complex128(_a.value), np.float64(_c.value), np.complex128(_s.value)


cublasZrotg.__doc__ = _ROTG_doc.substitute(
    precision="double precision",
    real="complex",
    type="numpy.complex128",
    c_type="numpy.float64",
    s_type="numpy.complex128",
    a_val="np.complex128(np.random.rand()+1j*np.random.rand())",
    b_val="np.complex128(np.random.rand()+1j*np.random.rand())",
    func="cublasZrotg",
)

# SROTM, DROTM (need to add example)
_ROTM_doc = Template(
    """
    Applies a real modified Givens rotation to two vectors.

    Applies the ${precision} real modified Givens rotation matrix `h`
    to the 2 x `n` matrix `[[x.T], [y.T]]`.

    Parameters
    ----------
    handle : int
        CUBLAS context.
    n : int
        Number of elements in input vectors.
    x : ctypes.c_void_p
        Pointer to the first input/output vector.
    incx : int
        Storage spacing between elements of `x`.
    y : ctypes.c_void_p
        Pointer to the second input/output vector.
    incy : int
        Storage spacing between elements of `y`.
    sparam : numpy.ndarray
        Array containing rotation parameters:
        sparam[0] contains the `flag`;
        sparam[1:5] contains the values `[h00, h10, h01, h11]`
        that determine the rotation matrix `h`.

    Notes
    -----
    The rotation matrix may assume the following values based on `flag`:

    - `flag` == -1.0: `h` == `[[h00, h01], [h10, h11]]`
    - `flag` == 0.0:  `h` == `[[1.0, h01], [h10, 1.0]]`
    - `flag` == 1.0:  `h` == `[[h00, 1.0], [-1.0, h11]]`
    - `flag` == -2.0: `h` == `[[1.0, 0.0], [0.0, 1.0]]` (identity)

    Both `x` and `y` must contain `n` elements.

    References
    ----------
    `cublas<t>srotm <http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-rotm>`_
"""
)

_libcublas.cublasSrotm_v2.restype = int
_libcublas.cublasSrotm_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]


def cublasSrotm(handle, n, x, incx, y, incy, sparam):
    """Applies a real modified Givens rotation (single precision)."""
    assert _libcublas
    status = _libcublas.cublasSrotm_v2(handle, n, int(x), incx, int(y), incy, int(sparam.ctypes.data))
    cublasCheckStatus(status)


cublasSrotm.__doc__ = _ROTM_doc.substitute(precision="single precision")

_libcublas.cublasDrotm_v2.restype = int
_libcublas.cublasDrotm_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]


def cublasDrotm(handle, n, x, incx, y, incy, sparam):
    """Applies a real modified Givens rotation (double precision)."""
    assert _libcublas
    status = _libcublas.cublasDrotm_v2(handle, n, int(x), incx, int(y), incy, int(sparam.ctypes.data))
    cublasCheckStatus(status)


cublasDrotm.__doc__ = _ROTM_doc.substitute(precision="double precision")

# SROTMG, DROTMG (need to add example)
_ROTMG_doc = Template(
    """
    Constructs a real modified Givens rotation matrix.

    Constructs the ${precision} real modified Givens rotation matrix
    `h = [[h11, h12], [h21, h22]]` that zeros out the second entry of
    the vector `[[sqrt(d1)*x1], [sqrt(d2)*x2]]`.

    Parameters
    ----------
    handle : int
        CUBLAS context.
    d1 : ${type}
        First scaling factor.
    d2 : ${type}
        Second scaling factor.
    x1 : ${type}
        First element of the input vector.
    y1 : ${type}
        Second element of the input vector.

    Returns
    -------
    sparam : numpy.ndarray
        Array containing rotation parameters:
        sparam[0] contains the `flag`;
        sparam[1:5] contains the values `[h00, h10, h01, h11]`
        that determine the rotation matrix `h`.

    Notes
    -----
    The rotation matrix may assume the following values based on `flag`:

    - `flag` == -1.0: `h` == `[[h00, h01], [h10, h11]]`
    - `flag` == 0.0:  `h` == `[[1.0, h01], [h10, 1.0]]`
    - `flag` == 1.0:  `h` == `[[h00, 1.0], [-1.0, h11]]`
    - `flag` == -2.0: `h` == `[[1.0, 0.0], [0.0, 1.0]]` (identity)

    References
    ----------
    `cublas<t>rotmg <http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-rotmg>`_
"""
)

_libcublas.cublasSrotmg_v2.restype = int
_libcublas.cublasSrotmg_v2.argtypes = [_types.handle, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]


def cublasSrotmg(handle, d1, d2, x1, y1):
    """Constructs a real modified Givens rotation matrix (single precision)."""
    _d1 = ctypes.c_float(d1)
    _d2 = ctypes.c_float(d2)
    _x1 = ctypes.c_float(x1)
    _y1 = ctypes.c_float(y1)
    sparam = np.empty(5, np.float32)
    assert _libcublas
    status = _libcublas.cublasSrotmg_v2(handle, ctypes.byref(_d1), ctypes.byref(_d2), ctypes.byref(_x1), ctypes.byref(_y1), int(sparam.ctypes.data))
    cublasCheckStatus(status)
    return sparam


cublasSrotmg.__doc__ = _ROTMG_doc.substitute(precision="single precision", type="numpy.float32")

_libcublas.cublasDrotmg_v2.restype = int
_libcublas.cublasDrotmg_v2.argtypes = [_types.handle, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]


def cublasDrotmg(handle, d1, d2, x1, y1):
    """Constructs a real modified Givens rotation matrix (double precision)."""
    _d1 = ctypes.c_double(d1)
    _d2 = ctypes.c_double(d2)
    _x1 = ctypes.c_double(x1)
    _y1 = ctypes.c_double(y1)
    sparam = np.empty(5, np.float64)
    assert _libcublas
    status = _libcublas.cublasDrotmg_v2(handle, ctypes.byref(_d1), ctypes.byref(_d2), ctypes.byref(_x1), ctypes.byref(_y1), int(sparam.ctypes.data))
    cublasCheckStatus(status)
    return sparam


cublasDrotmg.__doc__ = _ROTMG_doc.substitute(precision="double precision", type="numpy.float64")

# SSCAL, DSCAL, CSCAL, CSCAL, CSSCAL, ZSCAL, ZDSCAL
_SCAL_doc = Template(
    """
    Scales a vector by a scalar.

    Replaces a ${precision} ${real} vector `x` with `alpha * x`.

    Parameters
    ----------
    handle : int
        CUBLAS context.
    n : int
        Number of elements in input vector.
    alpha : ${a_type}
        Scalar multiplier.
    x : ctypes.c_void_p
        Pointer to the input/output vector.
    incx : int
        Storage spacing between elements of `x`.

    Examples
    --------
    >>> import pycuda.autoinit
    >>> import pycuda.gpuarray as gpuarray
    >>> import numpy as np
    >>> x = ${data}
    >>> x_gpu = gpuarray.to_gpu(x)
    >>> alpha = ${alpha}
    >>> h = cublasCreate()
    >>> ${func}(h, x.size, alpha, x_gpu.gpudata, 1)
    >>> cublasDestroy(h)
    >>> np.allclose(x_gpu.get(), alpha*x)
    True

    References
    ----------
    `cublas<t>scal <http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-scal>`_
"""
)

_libcublas.cublasSscal_v2.restype = int
_libcublas.cublasSscal_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]


def cublasSscal(handle, n, alpha, x, incx):
    """Scales a vector by a scalar (single precision real)."""
    assert _libcublas
    status = _libcublas.cublasSscal_v2(handle, n, ctypes.byref(ctypes.c_float(alpha)), int(x), incx)
    cublasCheckStatus(status)


cublasSscal.__doc__ = _SCAL_doc.substitute(
    precision="single precision", real="real", a_real="real", a_type="numpy.float32", alpha="np.float32(np.random.rand())", data="np.random.rand(5).astype(np.float32)", func="cublasSscal"
)

_libcublas.cublasDscal_v2.restype = int
_libcublas.cublasDscal_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]


def cublasDscal(handle, n, alpha, x, incx):
    """Scales a vector by a scalar (double precision real)."""
    assert _libcublas
    status = _libcublas.cublasDscal_v2(handle, n, ctypes.byref(ctypes.c_double(alpha)), int(x), incx)
    cublasCheckStatus(status)


cublasDscal.__doc__ = _SCAL_doc.substitute(
    precision="double precision", real="real", a_real="real", a_type="numpy.float64", alpha="np.float64(np.random.rand())", data="np.random.rand(5).astype(np.float64)", func="cublasDscal"
)

_libcublas.cublasCscal_v2.restype = int
_libcublas.cublasCscal_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]


def cublasCscal(handle, n, alpha, x, incx):
    """Scales a vector by a scalar (single precision complex)."""
    assert _libcublas
    status = _libcublas.cublasCscal_v2(handle, n, ctypes.byref(cuda.cuFloatComplex(alpha.real, alpha.imag)), int(x), incx)
    cublasCheckStatus(status)


cublasCscal.__doc__ = _SCAL_doc.substitute(
    precision="single precision",
    real="complex",
    a_real="complex",
    a_type="numpy.complex64",
    alpha="np.complex64(np.random.rand()+1j*np.random.rand())",
    data="(np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex64)",
    func="cublasCscal",
)

_libcublas.cublasCsscal_v2.restype = int
_libcublas.cublasCsscal_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]


def cublasCsscal(handle, n, alpha, x, incx):
    """Scales a complex vector by a real scalar (single precision)."""
    assert _libcublas
    status = _libcublas.cublasCsscal_v2(handle, n, ctypes.byref(ctypes.c_float(alpha)), int(x), incx)
    cublasCheckStatus(status)


cublasCsscal.__doc__ = _SCAL_doc.substitute(
    precision="single precision",
    real="complex",
    a_real="real",
    a_type="numpy.float32",
    alpha="np.float32(np.random.rand())",
    data="(np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex64)",
    func="cublasCsscal",
)

_libcublas.cublasZscal_v2.restype = int
_libcublas.cublasZscal_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]


def cublasZscal(handle, n, alpha, x, incx):
    """Scales a vector by a scalar (double precision complex)."""
    assert _libcublas
    status = _libcublas.cublasZscal_v2(handle, n, ctypes.byref(cuda.cuDoubleComplex(alpha.real, alpha.imag)), int(x), incx)
    cublasCheckStatus(status)


cublasZscal.__doc__ = _SCAL_doc.substitute(
    precision="double precision",
    real="complex",
    a_real="complex",
    a_type="numpy.complex128",
    alpha="np.complex128(np.random.rand()+1j*np.random.rand())",
    data="(np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex128)",
    func="cublasZscal",
)

_libcublas.cublasZdscal_v2.restype = int
_libcublas.cublasZdscal_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]


def cublasZdscal(handle, n, alpha, x, incx):
    """Scales a complex vector by a real scalar (double precision)."""
    assert _libcublas
    status = _libcublas.cublasZdscal_v2(handle, n, ctypes.byref(ctypes.c_double(alpha)), int(x), incx)
    cublasCheckStatus(status)


cublasZdscal.__doc__ = _SCAL_doc.substitute(
    precision="double precision",
    real="complex",
    a_real="real",
    a_type="numpy.float64",
    alpha="np.float64(np.random.rand())",
    data="(np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex128)",
    func="cublasZdscal",
)

# SSWAP, DSWAP, CSWAP, ZSWAP
_SWAP_doc = Template(
    """
    Swaps the contents of two vectors.

    Swaps the contents of one ${precision} ${real} vector with those
    of another ${precision} ${real} vector.

    Parameters
    ----------
    handle : int
        CUBLAS context.
    n : int
        Number of elements in input vectors.
    x : ctypes.c_void_p
        Pointer to the first input/output vector.
    incx : int
        Storage spacing between elements of `x`.
    y : ctypes.c_void_p
        Pointer to the second input/output vector.
    incy : int
        Storage spacing between elements of `y`.

    Examples
    --------
    >>> import pycuda.autoinit
    >>> import pycuda.gpuarray as gpuarray
    >>> import numpy as np
    >>> x = ${data}
    >>> y = ${data}
    >>> x_gpu = gpuarray.to_gpu(x)
    >>> y_gpu = gpuarray.to_gpu(y)
    >>> h = cublasCreate()
    >>> ${func}(h, x.size, x_gpu.gpudata, 1, y_gpu.gpudata, 1)
    >>> cublasDestroy(h)
    >>> np.allclose(x_gpu.get(), y)
    True
    >>> np.allclose(y_gpu.get(), x)
    True

    Notes
    -----
    Both `x` and `y` must contain `n` elements.

    References
    ----------
    `cublas<t>swap <http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-swap>`_
"""
)

_libcublas.cublasSswap_v2.restype = int
_libcublas.cublasSswap_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]


def cublasSswap(handle, n, x, incx, y, incy):
    """Swaps the contents of two vectors (single precision real)."""
    assert _libcublas
    status = _libcublas.cublasSswap_v2(handle, n, int(x), incx, int(y), incy)
    cublasCheckStatus(status)


cublasSswap.__doc__ = _SWAP_doc.substitute(precision="single precision", real="real", data="np.random.rand(5).astype(np.float32)", func="cublasSswap")

_libcublas.cublasDswap_v2.restype = int
_libcublas.cublasDswap_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]


def cublasDswap(handle, n, x, incx, y, incy):
    """Swaps the contents of two vectors (double precision real)."""
    assert _libcublas
    status = _libcublas.cublasDswap_v2(handle, n, int(x), incx, int(y), incy)
    cublasCheckStatus(status)


cublasDswap.__doc__ = _SWAP_doc.substitute(precision="double precision", real="real", data="np.random.rand(5).astype(np.float64)", func="cublasDswap")

_libcublas.cublasCswap_v2.restype = int
_libcublas.cublasCswap_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]


def cublasCswap(handle, n, x, incx, y, incy):
    """Swaps the contents of two vectors (single precision complex)."""
    assert _libcublas
    status = _libcublas.cublasCswap_v2(handle, n, int(x), incx, int(y), incy)
    cublasCheckStatus(status)


cublasCswap.__doc__ = _SWAP_doc.substitute(precision="single precision", real="complex", data="(np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex64)", func="cublasCswap")

_libcublas.cublasZswap_v2.restype = int
_libcublas.cublasZswap_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]


def cublasZswap(handle, n, x, incx, y, incy):
    """Swaps the contents of two vectors (double precision complex)."""
    assert _libcublas
    status = _libcublas.cublasZswap_v2(handle, n, int(x), incx, int(y), incy)
    cublasCheckStatus(status)


cublasZswap.__doc__ = _SWAP_doc.substitute(precision="double precision", real="complex", data="(np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex128)", func="cublasZswap")

# BLAS Level 2 Functions

# SGBMV, DGVMV, CGBMV, ZGBMV
_libcublas.cublasSgbmv_v2.restype = int
_libcublas.cublasSgbmv_v2.argtypes = [
    _types.handle,
    ctypes.c_char,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
]


def cublasSgbmv(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy):
    """
    Matrix-vector product for real single precision general banded matrix.

    Computes the product `alpha*op(A)*x + beta*y`, where `op(A)` is `A` or `A^T`
    or `A^H`, and `A` is a banded matrix.

    References
    ----------
    `cublas<t>gbmv <http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gbmv>`_
    """

    trans = trans.encode("ascii")
    assert _libcublas
    status = _libcublas.cublasSgbmv_v2(handle, trans, m, n, kl, ku, ctypes.byref(ctypes.c_float(alpha)), int(A), lda, int(x), incx, ctypes.byref(ctypes.c_float(beta)), int(y), incy)
    cublasCheckStatus(status)


_libcublas.cublasDgbmv_v2.restype = int
_libcublas.cublasDgbmv_v2.argtypes = [
    _types.handle,
    ctypes.c_char,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
]


def cublasDgbmv(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy):
    """
    Matrix-vector product for real double precision general banded matrix.

    Computes the product `alpha*op(A)*x + beta*y`, where `op(A)` is `A` or `A^T`
    or `A^H`, and `A` is a banded matrix.

    References
    ----------
    `cublas<t>gbmv <http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gbmv>`_
    """

    trans = trans.encode("ascii")
    assert _libcublas
    status = _libcublas.cublasDgbmv_v2(handle, trans, m, n, kl, ku, ctypes.byref(ctypes.c_float(alpha)), int(A), lda, int(x), incx, ctypes.byref(ctypes.c_float(beta)), int(y), incy)
    cublasCheckStatus(status)


_libcublas.cublasCgbmv_v2.restype = int
_libcublas.cublasCgbmv_v2.argtypes = [
    _types.handle,
    ctypes.c_char,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
]


def cublasCgbmv(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy):
    """
    Matrix-vector product for complex single precision general banded matrix.

    Computes the product `alpha*op(A)*x + beta*y`, where `op(A)` is `A` or `A^T`
    or `A^H`, and `A` is a banded matrix.

    References
    ----------
    `cublas<t>gbmv <http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gbmv>`_
    """

    trans = trans.encode("ascii")
    assert _libcublas
    status = _libcublas.cublasCgbmv_v2(
        handle, trans, m, n, kl, ku, ctypes.byref(cuda.cuFloatComplex(alpha.real, alpha.imag)), int(A), lda, int(x), incx, ctypes.byref(cuda.cuFloatComplex(beta.real, beta.imag)), int(y), incy
    )
    cublasCheckStatus(status)


_libcublas.cublasZgbmv_v2.restype = int
_libcublas.cublasZgbmv_v2.argtypes = [
    ctypes.c_char,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
]


def cublasZgbmv(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy):
    """
    Matrix-vector product for complex double precision general banded matrix.

    Computes the product `alpha*op(A)*x + beta*y`, where `op(A)` is `A` or `A^T`
    or `A^H`, and `A` is a banded matrix.

    References
    ----------
    `cublas<t>gbmv <http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gbmv>`_
    """
    trans = trans.encode("ascii")
    assert _libcublas
    status = _libcublas.cublasZgbmv_v2(
        handle, trans, m, n, kl, ku, ctypes.byref(cuda.cuDoubleComplex(alpha.real, alpha.imag)), int(A), lda, int(x), incx, ctypes.byref(cuda.cuDoubleComplex(beta.real, beta.imag)), int(y), incy
    )
    cublasCheckStatus(status)


# SGEMV, DGEMV, CGEMV, ZGEMV # XXX need to adjust
_GEMV_doc = Template(
    """
    Matrix-vector product for ${precision} ${real} general matrix.

    Computes the product `alpha*op(A)*x + beta*y`, where `op(A)` is `A`
    or `A.T` or `A.conj().T`, and stores the result in `y`.

    Parameters
    ----------
    handle : int
        CUBLAS context.
    trans : char
        Specifies the form of the matrix `A`.
        'N' or 'n': `A` is not transposed or conjugated.
        'T' or 't': `A` is transposed.
        'C' or 'c': `A` is transposed and conjugated.
    m : int
        Number of rows of matrix `A`.
    n : int
        Number of columns of matrix `A`.
    alpha : ${a_type}
        Scalar multiplier for matrix `A`.
    A : ctypes.c_void_p
        Pointer to the matrix `A`.
        Shape is `(lda, n)` if `trans` is 'N' or 'n'.
        Shape is `(lda, m)` if `trans` is 'T', 't', 'C', or 'c'.
    lda : int
        Leading dimension of `A`.
    x : ctypes.c_void_p
        Pointer to the vector `x`.
    incx : int
        Storage spacing between elements of `x`.
    beta : ${a_type}
        Scalar multiplier for vector `y`.
    y : ctypes.c_void_p
        Pointer to the vector `y`. The result is stored here.
    incy : int
        Storage spacing between elements of `y`.

    References
    ----------
    `cublas<t>gemv <http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemv>`_
"""
)

_libcublas.cublasSgemv_v2.restype = int
_libcublas.cublasSgemv_v2.argtypes = [
    _types.handle,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
]


def cublasSgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy):
    """Matrix-vector product (single precision real)."""
    assert _libcublas
    status = _libcublas.cublasSgemv_v2(handle, _CUBLAS_OP[trans], m, n, ctypes.byref(ctypes.c_float(alpha)), int(A), lda, int(x), incx, ctypes.byref(ctypes.c_float(beta)), int(y), incy)
    cublasCheckStatus(status)


cublasSgemv.__doc__ = _GEMV_doc.substitute(precision="single precision", real="real", a_type="numpy.float32")

_libcublas.cublasDgemv_v2.restype = int
_libcublas.cublasDgemv_v2.argtypes = [
    _types.handle,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
]


def cublasDgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy):
    """Matrix-vector product (double precision real)."""
    assert _libcublas
    status = _libcublas.cublasDgemv_v2(handle, _CUBLAS_OP[trans], m, n, ctypes.byref(ctypes.c_double(alpha)), int(A), lda, int(x), incx, ctypes.byref(ctypes.c_double(beta)), int(y), incy)
    cublasCheckStatus(status)


cublasDgemv.__doc__ = _GEMV_doc.substitute(precision="double precision", real="real", a_type="numpy.float64")

_libcublas.cublasCgemv_v2.restype = int
_libcublas.cublasCgemv_v2.argtypes = [
    _types.handle,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
]


def cublasCgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy):
    """Matrix-vector product (single precision complex)."""
    assert _libcublas
    status = _libcublas.cublasCgemv_v2(
        handle, _CUBLAS_OP[trans], m, n, ctypes.byref(cuda.cuFloatComplex(alpha.real, alpha.imag)), int(A), lda, int(x), incx, ctypes.byref(cuda.cuFloatComplex(beta.real, beta.imag)), int(y), incy
    )
    cublasCheckStatus(status)


cublasCgemv.__doc__ = _GEMV_doc.substitute(precision="single precision", real="complex", a_type="numpy.complex64")

_libcublas.cublasZgemv_v2.restype = int
_libcublas.cublasZgemv_v2.argtypes = [
    _types.handle,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
]


def cublasZgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy):
    """Matrix-vector product (double precision complex)."""
    assert _libcublas
    status = _libcublas.cublasZgemv_v2(
        handle, _CUBLAS_OP[trans], m, n, ctypes.byref(cuda.cuDoubleComplex(alpha.real, alpha.imag)), int(A), lda, int(x), incx, ctypes.byref(cuda.cuDoubleComplex(beta.real, beta.imag)), int(y), incy
    )
    cublasCheckStatus(status)


cublasZgemv.__doc__ = _GEMV_doc.substitute(precision="double precision", real="complex", a_type="numpy.complex128")


# SGER, DGER, CGERU, CGERC, ZGERU, ZGERC
_GER_doc = Template(
    """
    Rank-1 operation on a general matrix.

    Updates the matrix `A` with the rank-1 operation:
    `A = alpha*x*y^T + A` (for real) or `A = alpha*x*y^H + A` (for complex).

    Parameters
    ----------
    handle : int
        CUBLAS context.
    m : int
        Number of rows of matrix `A`.
    n : int
        Number of columns of matrix `A`.
    alpha : ${a_type}
        Scalar multiplier.
    x : ctypes.c_void_p
        Pointer to the vector `x`.
    incx : int
        Storage spacing between elements of `x`.
    y : ctypes.c_void_p
        Pointer to the vector `y`.
    incy : int
        Storage spacing between elements of `y`.
    A : ctypes.c_void_p
        Pointer to the matrix `A`.
    lda : int
        Leading dimension of `A`.

    References
    ----------
    `cublas<t>ger <http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-ger>`_
"""
)

_libcublas.cublasSger_v2.restype = int
_libcublas.cublasSger_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]


def cublasSger(handle, m, n, alpha, x, incx, y, incy, A, lda):
    """Rank-1 operation (single precision real)."""
    assert _libcublas
    status = _libcublas.cublasSger_v2(handle, m, n, ctypes.byref(ctypes.c_float(alpha)), int(x), incx, int(y), incy, int(A), lda)
    cublasCheckStatus(status)


cublasSger.__doc__ = _GER_doc.substitute(a_type="numpy.float32")

_libcublas.cublasDger_v2.restype = int
_libcublas.cublasDger_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]


def cublasDger(handle, m, n, alpha, x, incx, y, incy, A, lda):
    """Rank-1 operation (double precision real)."""
    assert _libcublas
    status = _libcublas.cublasDger_v2(handle, m, n, ctypes.byref(ctypes.c_double(alpha)), int(x), incx, int(y), incy, int(A), lda)
    cublasCheckStatus(status)


cublasDger.__doc__ = _GER_doc.substitute(a_type="numpy.float64")

_libcublas.cublasCgerc_v2.restype = int
_libcublas.cublasCgerc_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]


def cublasCgerc(handle, m, n, alpha, x, incx, y, incy, A, lda):
    """Rank-1 operation (single precision complex, conjugate)."""
    assert _libcublas
    status = _libcublas.cublasCgerc_v2(handle, m, n, ctypes.byref(cuda.cuFloatComplex(alpha.real, alpha.imag)), int(x), incx, int(y), incy, int(A), lda)
    cublasCheckStatus(status)


cublasCgerc.__doc__ = _GER_doc.substitute(a_type="numpy.complex64")

_libcublas.cublasCgeru_v2.restype = int
_libcublas.cublasCgeru_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]


def cublasCgeru(handle, m, n, alpha, x, incx, y, incy, A, lda):
    """Rank-1 operation (single precision complex, non-conjugate)."""
    assert _libcublas
    status = _libcublas.cublasCgeru_v2(handle, m, n, ctypes.byref(cuda.cuFloatComplex(alpha.real, alpha.imag)), int(x), incx, int(y), incy, int(A), lda)
    cublasCheckStatus(status)


cublasCgeru.__doc__ = _GER_doc.substitute(a_type="numpy.complex64")

_libcublas.cublasZgerc_v2.restype = int
_libcublas.cublasZgerc_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]


def cublasZgerc(handle, m, n, alpha, x, incx, y, incy, A, lda):
    """Rank-1 operation (double precision complex, conjugate)."""
    assert _libcublas
    status = _libcublas.cublasZgerc_v2(handle, m, n, ctypes.byref(cuda.cuDoubleComplex(alpha.real, alpha.imag)), int(x), incx, int(y), incy, int(A), lda)
    cublasCheckStatus(status)


cublasZgerc.__doc__ = _GER_doc.substitute(a_type="numpy.complex128")

_libcublas.cublasZgeru_v2.restype = int
_libcublas.cublasZgeru_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]


def cublasZgeru(handle, m, n, alpha, x, incx, y, incy, A, lda):
    """Rank-1 operation (double precision complex, non-conjugate)."""
    assert _libcublas
    status = _libcublas.cublasZgeru_v2(handle, m, n, ctypes.byref(cuda.cuDoubleComplex(alpha.real, alpha.imag)), int(x), incx, int(y), incy, int(A), lda)
    cublasCheckStatus(status)


cublasZgeru.__doc__ = _GER_doc.substitute(a_type="numpy.complex128")


# SSBMV, DSBMV
_SBMV_doc = Template(
    """
    Matrix-vector product for ${precision} ${real} symmetric-banded matrix.

    Computes the product `alpha*A*x + beta*y`, where `A` is a symmetric-banded
    matrix.

    Parameters
    ----------
    handle : int
        CUBLAS context.
    uplo : char
        Specifies whether `A` is upper or lower triangular:
        'U' or 'u': `A` is upper banded.
        'L' or 'l': `A` is lower banded.
    n : int
        Number of columns of `A`.
    k : int
        Number of super- or sub-diagonals of `A`.
    alpha : ${a_type}
        Scalar multiplier for matrix `A`.
    A : ctypes.c_void_p
        Pointer to the matrix `A`.
    lda : int
        Leading dimension of `A`.
    x : ctypes.c_void_p
        Pointer to the vector `x`.
    incx : int
        Storage spacing between elements of `x`.
    beta : ${a_type}
        Scalar multiplier for vector `y`.
    y : ctypes.c_void_p
        Pointer to the vector `y`. The result is stored here.
    incy : int
        Storage spacing between elements of `y`.

    References
    ----------
    `cublas<t>sbmv <http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-sbmv>`_
    """
)

_libcublas.cublasSsbmv_v2.restype = int
_libcublas.cublasSsbmv_v2.argtypes = [
    _types.handle,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
]


def cublasSsbmv(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy):
    """Matrix-vector product (single precision real symmetric-banded)."""
    assert _libcublas
    status = _libcublas.cublasSsbmv_v2(handle, _CUBLAS_FILL_MODE[uplo], n, k, ctypes.byref(ctypes.c_float(alpha)), int(A), lda, int(x), incx, ctypes.byref(ctypes.c_float(beta)), int(y), incy)
    cublasCheckStatus(status)


cublasSsbmv.__doc__ = _SBMV_doc.substitute(precision="single precision", real="real", a_type="numpy.float32")

_libcublas.cublasDsbmv_v2.restype = int
_libcublas.cublasDsbmv_v2.argtypes = [
    _types.handle,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
]


def cublasDsbmv(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy):
    """Matrix-vector product (double precision real symmetric-banded)."""
    assert _libcublas
    status = _libcublas.cublasDsbmv_v2(handle, _CUBLAS_FILL_MODE[uplo], n, k, ctypes.byref(ctypes.c_double(alpha)), int(A), lda, int(x), incx, ctypes.byref(ctypes.c_double(beta)), int(y), incy)
    cublasCheckStatus(status)


cublasDsbmv.__doc__ = _SBMV_doc.substitute(precision="double precision", real="real", a_type="numpy.float64")


# SSPMV, DSPMV
_SPMV_doc = Template(
    """
    Matrix-vector product for ${precision} ${real} symmetric packed matrix.

    Computes the product `alpha*AP*x + beta*y`, where `AP` is a symmetric packed
    matrix.

    Parameters
    ----------
    handle : int
        CUBLAS context.
    uplo : char
        Specifies whether `AP` is upper or lower triangular:
        'U' or 'u': `AP` is upper packed.
        'L' or 'l': `AP` is lower packed.
    n : int
        Number of elements of the matrix `AP`.
    alpha : ${a_type}
        Scalar multiplier for matrix `AP`.
    AP : ctypes.c_void_p
        Pointer to the packed symmetric matrix `AP`.
    x : ctypes.c_void_p
        Pointer to the vector `x`.
    incx : int
        Storage spacing between elements of `x`.
    beta : ${a_type}
        Scalar multiplier for vector `y`.
    y : ctypes.c_void_p
        Pointer to the vector `y`. The result is stored here.
    incy : int
        Storage spacing between elements of `y`.

    References
    ----------
    `cublas<t>spmv <http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-spmv>`_
    """
)

_libcublas.cublasSspmv_v2.restype = int
_libcublas.cublasSspmv_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]


def cublasSspmv(handle, uplo, n, alpha, AP, x, incx, beta, y, incy):
    """Matrix-vector product (single precision real symmetric packed)."""
    assert _libcublas
    status = _libcublas.cublasSspmv_v2(
        handle, _CUBLAS_FILL_MODE[uplo], n, ctypes.byref(ctypes.c_float(alpha)), ctypes.byref(ctypes.c_float(AP)), int(x), incx, ctypes.byref(ctypes.c_float(beta)), int(y), incy
    )
    cublasCheckStatus(status)


cublasSspmv.__doc__ = _SPMV_doc.substitute(precision="single precision", real="real", a_type="numpy.float32")

_libcublas.cublasDspmv_v2.restype = int
_libcublas.cublasDspmv_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]


def cublasDspmv(handle, uplo, n, alpha, AP, x, incx, beta, y, incy):
    """Matrix-vector product (double precision real symmetric packed)."""
    assert _libcublas
    status = _libcublas.cublasDspmv_v2(
        handle, _CUBLAS_FILL_MODE[uplo], n, ctypes.byref(ctypes.c_double(alpha)), ctypes.byref(ctypes.c_double(AP)), int(x), incx, ctypes.byref(ctypes.c_double(beta)), int(y), incy
    )
    cublasCheckStatus(status)


cublasDspmv.__doc__ = _SPMV_doc.substitute(precision="double precision", real="real", a_type="numpy.float64")


# SSPR, DSPR
_SPR_doc = Template(
    """
    Rank-1 operation on a symmetric packed matrix.

    Updates the symmetric packed matrix `AP` with a rank-1 operation:
    `AP = alpha*x*x^T + AP` (for real) or `AP = alpha*x*x^H + AP` (for complex).

    Parameters
    ----------
    handle : int
        CUBLAS context.
    uplo : char
        Specifies whether `AP` is upper or lower triangular:
        'U' or 'u': `AP` is upper packed.
        'L' or 'l': `AP` is lower packed.
    n : int
        Number of elements of the matrix `AP`.
    alpha : ${a_type}
        Scalar multiplier.
    x : ctypes.c_void_p
        Pointer to the vector `x`.
    incx : int
        Storage spacing between elements of `x`.
    AP : ctypes.c_void_p
        Pointer to the packed symmetric matrix `AP`.

    References
    ----------
    `cublas<t>spr <http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-spr>`_
    """
)

_libcublas.cublasSspr_v2.restype = int
_libcublas.cublasSspr_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]


def cublasSspr(handle, uplo, n, alpha, x, incx, AP):
    """Rank-1 operation (single precision real symmetric packed)."""
    assert _libcublas
    status = _libcublas.cublasSspr_v2(handle, _CUBLAS_FILL_MODE[uplo], n, ctypes.byref(ctypes.c_float(alpha)), int(x), incx, int(AP))
    cublasCheckStatus(status)


cublasSspr.__doc__ = _SPR_doc.substitute(a_type="numpy.float32")

_libcublas.cublasDspr_v2.restype = int
_libcublas.cublasDspr_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]


def cublasDspr(handle, uplo, n, alpha, x, incx, AP):
    """Rank-1 operation (double precision real symmetric packed)."""
    assert _libcublas
    status = _libcublas.cublasDspr_v2(handle, _CUBLAS_FILL_MODE[uplo], n, ctypes.byref(ctypes.c_double(alpha)), int(x), incx, int(AP))
    cublasCheckStatus(status)


cublasDspr.__doc__ = _SPR_doc.substitute(a_type="numpy.float64")


# SSPR2, DSPR2
_SPR2_doc = Template(
    """
    Rank-2 operation on a symmetric packed matrix.

    Updates the symmetric packed matrix `AP` with a rank-2 operation:
    `AP = alpha*x*y^T + conj(alpha)*y*x^T + AP` (for complex) or
    `AP = alpha*x*y^T + alpha*y*x^T + AP` (for real).

    Parameters
    ----------
    handle : int
        CUBLAS context.
    uplo : char
        Specifies whether `AP` is upper or lower triangular:
        'U' or 'u': `AP` is upper packed.
        'L' or 'l': `AP` is lower packed.
    n : int
        Number of elements of the matrix `AP`.
    alpha : ${a_type}
        Scalar multiplier.
    x : ctypes.c_void_p
        Pointer to the first vector `x`.
    incx : int
        Storage spacing between elements of `x`.
    y : ctypes.c_void_p
        Pointer to the second vector `y`.
    incy : int
        Storage spacing between elements of `y`.
    AP : ctypes.c_void_p
        Pointer to the packed symmetric matrix `AP`.

    References
    ----------
    `cublas<t>spr2 <http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-spr2>`_
    """
)

_libcublas.cublasSspr2_v2.restype = int
_libcublas.cublasSspr2_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]


def cublasSspr2(handle, uplo, n, alpha, x, incx, y, incy, AP):
    """Rank-2 operation (single precision real symmetric packed)."""
    assert _libcublas
    status = _libcublas.cublasSspr2_v2(handle, _CUBLAS_FILL_MODE[uplo], n, ctypes.byref(ctypes.c_float(alpha)), int(x), incx, int(y), incy, int(AP))

    cublasCheckStatus(status)


cublasSspr2.__doc__ = _SPR2_doc.substitute(a_type="numpy.float32")

_libcublas.cublasDspr2_v2.restype = int
_libcublas.cublasDspr2_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]


def cublasDspr2(handle, uplo, n, alpha, x, incx, y, incy, AP):
    """Rank-2 operation (double precision real symmetric packed)."""
    assert _libcublas
    status = _libcublas.cublasDspr2_v2(handle, _CUBLAS_FILL_MODE[uplo], n, ctypes.byref(ctypes.c_double(alpha)), int(x), incx, int(y), incy, int(AP))
    cublasCheckStatus(status)


cublasDspr2.__doc__ = _SPR2_doc.substitute(a_type="numpy.float64")


# SSYMV, DSYMV, CSYMV, ZSYMV
_SYMV_doc = Template(
    """
    Matrix-vector product for a symmetric matrix.

    Computes the product `alpha*A*x + beta*y`, where `A` is a symmetric matrix.

    Parameters
    ----------
    handle : int
        CUBLAS context.
    uplo : char
        Specifies whether `A` is upper or lower triangular:
        'U' or 'u': `A` is upper triangular.
        'L' or 'l': `A` is lower triangular.
    n : int
        Number of columns of `A`.
    alpha : ${a_type}
        Scalar multiplier for matrix `A`.
    A : ctypes.c_void_p
        Pointer to the matrix `A`.
    lda : int
        Leading dimension of `A`.
    x : ctypes.c_void_p
        Pointer to the vector `x`.
    incx : int
        Storage spacing between elements of `x`.
    beta : ${a_type}
        Scalar multiplier for vector `y`.
    y : ctypes.c_void_p
        Pointer to the vector `y`. The result is stored here.
    incy : int
        Storage spacing between elements of `y`.

    References
    ----------
    `cublas<t>symv <http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-symv>`_
    """
)

_libcublas.cublasSsymv_v2.restype = int
_libcublas.cublasSsymv_v2.argtypes = [
    _types.handle,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
]


def cublasSsymv(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy):
    """Matrix-vector product (single precision real symmetric)."""
    assert _libcublas
    status = _libcublas.cublasSsymv_v2(handle, _CUBLAS_FILL_MODE[uplo], n, ctypes.byref(ctypes.c_float(alpha)), int(A), lda, int(x), incx, ctypes.byref(ctypes.c_float(beta)), int(y), incy)
    cublasCheckStatus(status)


cublasSsymv.__doc__ = _SYMV_doc.substitute(precision="single precision", real="real", a_type="numpy.float32")

_libcublas.cublasDsymv_v2.restype = int
_libcublas.cublasDsymv_v2.argtypes = [
    _types.handle,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
]


def cublasDsymv(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy):
    """Matrix-vector product (double precision real symmetric)."""
    assert _libcublas
    status = _libcublas.cublasDsymv_v2(handle, _CUBLAS_FILL_MODE[uplo], n, ctypes.byref(ctypes.c_double(alpha)), int(A), lda, int(x), incx, ctypes.byref(ctypes.c_double(beta)), int(y), incy)
    cublasCheckStatus(status)


cublasDsymv.__doc__ = _SYMV_doc.substitute(precision="double precision", real="real", a_type="numpy.float64")


if _cublas_version >= 5000:
    _libcublas.cublasCsymv_v2.restype = int
    _libcublas.cublasCsymv_v2.argtypes = [
        _types.handle,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
    ]


@_cublas_version_req(5.0)
def cublasCsymv(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy):
    """Matrix-vector product (single precision complex symmetric)."""
    assert _libcublas
    status = _libcublas.cublasCsymv_v2(
        handle, _CUBLAS_FILL_MODE[uplo], n, ctypes.byref(cuda.cuFloatComplex(alpha.real, alpha.imag)), int(A), lda, int(x), incx, ctypes.byref(cuda.cuFloatComplex(beta.real, beta.imag)), int(y), incy
    )
    cublasCheckStatus(status)


if _cublas_version >= 5000:
    _libcublas.cublasZsymv_v2.restype = int
    _libcublas.cublasZsymv_v2.argtypes = [
        _types.handle,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
    ]


@_cublas_version_req(5.0)
def cublasZsymv(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy):
    """Matrix-vector product (double precision complex symmetric)."""
    assert _libcublas
    status = _libcublas.cublasZsymv_v2(
        handle,
        _CUBLAS_FILL_MODE[uplo],
        n,
        ctypes.byref(cuda.cuDoubleComplex(alpha.real, alpha.imag)),
        int(A),
        lda,
        int(x),
        incx,
        ctypes.byref(cuda.cuDoubleComplex(beta.real, beta.imag)),
        int(y),
        incy,
    )
    cublasCheckStatus(status)


# SSYR, DSYR, CSYR, ZSYR
_SYR_doc = Template(
    """
    Rank-1 operation on a symmetric matrix.

    Updates the symmetric matrix `A` with a rank-1 operation:
    `A = alpha*x*x^T + A` (for real) or `A = alpha*x*x^H + A` (for complex).

    Parameters
    ----------
    handle : int
        CUBLAS context.
    uplo : char
        Specifies whether `A` is upper or lower triangular:
        'U' or 'u': `A` is upper triangular.
        'L' or 'l': `A` is lower triangular.
    n : int
        Number of columns of `A`.
    alpha : ${a_type}
        Scalar multiplier.
    x : ctypes.c_void_p
        Pointer to the vector `x`.
    incx : int
        Storage spacing between elements of `x`.
    A : ctypes.c_void_p
        Pointer to the matrix `A`.
    lda : int
        Leading dimension of `A`.

    References
    ----------
    `cublas<t>syr <http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-syr>`_
    """
)

_libcublas.cublasSsyr_v2.restype = int
_libcublas.cublasSsyr_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]


def cublasSsyr(handle, uplo, n, alpha, x, incx, A, lda):
    """Rank-1 operation (single precision real symmetric)."""
    assert _libcublas
    status = _libcublas.cublasSsyr_v2(handle, _CUBLAS_FILL_MODE[uplo], n, ctypes.byref(ctypes.c_float(alpha)), int(x), incx, int(A), lda)
    cublasCheckStatus(status)


cublasSsyr.__doc__ = _SYR_doc.substitute(a_type="numpy.float32")

_libcublas.cublasDsyr_v2.restype = int
_libcublas.cublasDsyr_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]


def cublasDsyr(handle, uplo, n, alpha, x, incx, A, lda):
    """Rank-1 operation (double precision real symmetric)."""
    assert _libcublas
    status = _libcublas.cublasDsyr_v2(handle, _CUBLAS_FILL_MODE[uplo], n, ctypes.byref(ctypes.c_double(alpha)), int(x), incx, int(A), lda)
    cublasCheckStatus(status)


cublasDsyr.__doc__ = _SYR_doc.substitute(a_type="numpy.float64")


if _cublas_version >= 5000:
    _libcublas.cublasCsyr_v2.restype = int
    _libcublas.cublasCsyr_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]


@_cublas_version_req(5.0)
def cublasCsyr(handle, uplo, n, alpha, x, incx, A, lda):
    """Rank-1 operation (single precision complex symmetric)."""
    assert _libcublas
    status = _libcublas.cublasCsyr_v2(handle, _CUBLAS_FILL_MODE[uplo], n, ctypes.byref(cuda.cuFloatComplex(alpha.real, alpha.imag)), int(x), incx, int(A), lda)
    cublasCheckStatus(status)


if _cublas_version >= 5000:
    _libcublas.cublasZsyr_v2.restype = int
    _libcublas.cublasZsyr_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]


@_cublas_version_req(5.0)
def cublasZsyr(handle, uplo, n, alpha, x, incx, A, lda):
    """Rank-1 operation (double precision complex symmetric)."""
    assert _libcublas
    status = _libcublas.cublasZsyr_v2(handle, _CUBLAS_FILL_MODE[uplo], n, ctypes.byref(cuda.cuDoubleComplex(alpha.real, alpha.imag)), int(x), incx, int(A), lda)
    cublasCheckStatus(status)


# SSYR2, DSYR2, CSYR2, ZSYR2
_SYR2_doc = Template(
    """
    Rank-2 operation on a symmetric matrix.

    Updates the symmetric matrix `A` with a rank-2 operation:
    `A = alpha*x*y^T + conj(alpha)*y*x^T + A` (for complex) or
    `A = alpha*x*y^T + alpha*y*x^T + A` (for real).

    Parameters
    ----------
    handle : int
        CUBLAS context.
    uplo : char
        Specifies whether `A` is upper or lower triangular:
        'U' or 'u': `A` is upper triangular.
        'L' or 'l': `A` is lower triangular.
    n : int
        Number of columns of `A`.
    alpha : ${a_type}
        Scalar multiplier.
    x : ctypes.c_void_p
        Pointer to the first vector `x`.
    incx : int
        Storage spacing between elements of `x`.
    y : ctypes.c_void_p
        Pointer to the second vector `y`.
    incy : int
        Storage spacing between elements of `y`.
    A : ctypes.c_void_p
        Pointer to the matrix `A`.
    lda : int
        Leading dimension of `A`.

    References
    ----------
    `cublas<t>syr2 <http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-syr2>`_
    """
)

_libcublas.cublasSsyr2_v2.restype = int
_libcublas.cublasSsyr2_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]


def cublasSsyr2(handle, uplo, n, alpha, x, incx, y, incy, A, lda):
    """Rank-2 operation (single precision real symmetric)."""
    assert _libcublas
    status = _libcublas.cublasSsyr2_v2(handle, _CUBLAS_FILL_MODE[uplo], n, ctypes.byref(ctypes.c_float(alpha)), int(x), incx, int(y), incy, int(A), lda)
    cublasCheckStatus(status)


cublasSsyr2.__doc__ = _SYR2_doc.substitute(a_type="numpy.float32")

_libcublas.cublasDsyr2_v2.restype = int
_libcublas.cublasDsyr2_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]


def cublasDsyr2(handle, uplo, n, alpha, x, incx, y, incy, A, lda):
    """Rank-2 operation (double precision real symmetric)."""
    assert _libcublas
    status = _libcublas.cublasDsyr2_v2(handle, _CUBLAS_FILL_MODE[uplo], n, ctypes.byref(ctypes.c_double(alpha)), int(x), incx, int(y), incy, int(A), lda)
    cublasCheckStatus(status)


cublasDsyr2.__doc__ = _SYR2_doc.substitute(a_type="numpy.float64")


if _cublas_version >= 5000:
    _libcublas.cublasCsyr2_v2.restype = int
    _libcublas.cublasCsyr2_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]


@_cublas_version_req(5.0)
def cublasCsyr2(handle, uplo, n, alpha, x, incx, y, incy, A, lda):
    """Rank-2 operation (single precision complex symmetric)."""
    assert _libcublas
    status = _libcublas.cublasCsyr2_v2(handle, _CUBLAS_FILL_MODE[uplo], n, ctypes.byref(cuda.cuFloatComplex(alpha.real, alpha.imag)), int(x), incx, int(y), incy, int(A), lda)
    cublasCheckStatus(status)


if _cublas_version >= 5000:
    _libcublas.cublasZsyr2_v2.restype = int
    _libcublas.cublasZsyr2_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]


@_cublas_version_req(5.0)
def cublasZsyr2(handle, uplo, n, alpha, x, incx, y, incy, A, lda):
    """Rank-2 operation (double precision complex symmetric)."""
    assert _libcublas
    status = _libcublas.cublasZsyr2_v2(handle, _CUBLAS_FILL_MODE[uplo], n, ctypes.byref(cuda.cuDoubleComplex(alpha.real, alpha.imag)), int(x), incx, int(y), incy, int(A), lda)
    cublasCheckStatus(status)


# STBMV, DTBMV, CTBMV, ZTBMV
_TBMV_doc = Template(
    """
    Matrix-vector product for a triangular banded matrix.

    Computes the product `A*x`, where `A` is a triangular banded matrix.

    Parameters
    ----------
    handle : int
        CUBLAS context.
    uplo : char
        Specifies whether `A` is upper or lower triangular:
        'U' or 'u': `A` is upper triangular.
        'L' or 'l': `A` is lower triangular.
    trans : char
        Specifies the form of the matrix `A`:
        'N' or 'n': `A` is not transposed or conjugated.
        'T' or 't': `A` is transposed.
        'C' or 'c': `A` is transposed and conjugated.
    diag : char
        Specifies whether `A` is unit triangular:
        'U' or 'u': `A` is unit triangular.
        'N' or 'n': `A` is not unit triangular.
    n : int
        Number of columns of `A`.
    k : int
        Number of super- or sub-diagonals of `A`.
    A : ctypes.c_void_p
        Pointer to the matrix `A`.
    lda : int
        Leading dimension of `A`.
    x : ctypes.c_void_p
        Pointer to the vector `x`.
    incx : int
        Storage spacing between elements of `x`.

    References
    ----------
    `cublas<t>tbmv <http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-tbmv>`_
    """
)

_libcublas.cublasStbmv_v2.restype = int
_libcublas.cublasStbmv_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]


def cublasStbmv(handle, uplo, trans, diag, n, k, A, lda, x, incx):
    """Matrix-vector product (single precision real triangular banded)."""
    assert _libcublas
    status = _libcublas.cublasStbmv_v2(handle, _CUBLAS_FILL_MODE[uplo], _CUBLAS_OP[trans], _CUBLAS_DIAG[diag], n, k, int(A), lda, int(x), incx)
    cublasCheckStatus(status)


cublasStbmv.__doc__ = _TBMV_doc.substitute()

_libcublas.cublasDtbmv_v2.restype = int
_libcublas.cublasDtbmv_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]


def cublasDtbmv(handle, uplo, trans, diag, n, k, A, lda, x, incx):
    """Matrix-vector product (double precision real triangular banded)."""
    assert _libcublas
    status = _libcublas.cublasDtbmv_v2(handle, _CUBLAS_FILL_MODE[uplo], _CUBLAS_OP[trans], _CUBLAS_DIAG[diag], n, k, int(A), lda, int(x), incx)
    cublasCheckStatus(status)


cublasDtbmv.__doc__ = _TBMV_doc.substitute()

_libcublas.cublasCtbmv_v2.restype = int
_libcublas.cublasCtbmv_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]


def cublasCtbmv(handle, uplo, trans, diag, n, k, A, lda, x, incx):
    """Matrix-vector product (single precision complex triangular banded)."""
    assert _libcublas
    status = _libcublas.cublasCtbmv_v2(handle, _CUBLAS_FILL_MODE[uplo], _CUBLAS_OP[trans], _CUBLAS_DIAG[diag], n, k, int(A), lda, int(x), incx)
    cublasCheckStatus(status)


cublasCtbmv.__doc__ = _TBMV_doc.substitute()

_libcublas.cublasZtbmv_v2.restype = int
_libcublas.cublasZtbmv_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]


def cublasZtbmv(handle, uplo, trans, diag, n, k, A, lda, x, incx):
    """Matrix-vector product (double precision complex triangular banded)."""
    assert _libcublas
    status = _libcublas.cublasZtbmv_v2(handle, _CUBLAS_FILL_MODE[uplo], _CUBLAS_OP[trans], _CUBLAS_DIAG[diag], n, k, int(A), lda, int(x), incx)
    cublasCheckStatus(status)


cublasZtbmv.__doc__ = _TBMV_doc.substitute()


# STBSV, DTBSV, CTBSV, ZTBSV
_TBSV_doc = Template(
    """
    Solves a triangular banded system with one right-hand side.

    Solves the system `A*x = b`, where `A` is a triangular banded matrix.

    Parameters
    ----------
    handle : int
        CUBLAS context.
    uplo : char
        Specifies whether `A` is upper or lower triangular:
        'U' or 'u': `A` is upper triangular.
        'L' or 'l': `A` is lower triangular.
    trans : char
        Specifies the form of the matrix `A`:
        'N' or 'n': `A` is not transposed or conjugated.
        'T' or 't': `A` is transposed.
        'C' or 'c': `A` is transposed and conjugated.
    diag : char
        Specifies whether `A` is unit triangular:
        'U' or 'u': `A` is unit triangular.
        'N' or 'n': `A` is not unit triangular.
    n : int
        Number of columns of `A`.
    k : int
        Number of super- or sub-diagonals of `A`.
    A : ctypes.c_void_p
        Pointer to the matrix `A`.
    lda : int
        Leading dimension of `A`.
    x : ctypes.c_void_p
        Pointer to the vector `x`. The solution is stored here.
    incx : int
        Storage spacing between elements of `x`.

    References
    ----------
    `cublas<t>tbsv <http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-tbsv>`_
    """
)

_libcublas.cublasStbsv_v2.restype = int
_libcublas.cublasStbsv_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]


def cublasStbsv(handle, uplo, trans, diag, n, k, A, lda, x, incx):
    """Solves a triangular banded system (single precision real)."""
    assert _libcublas
    status = _libcublas.cublasStbsv_v2(handle, _CUBLAS_FILL_MODE[uplo], _CUBLAS_OP[trans], _CUBLAS_DIAG[diag], n, k, int(A), lda, int(x), incx)
    cublasCheckStatus(status)


cublasStbsv.__doc__ = _TBSV_doc.substitute()

_libcublas.cublasDtbsv_v2.restype = int
_libcublas.cublasDtbsv_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]


def cublasDtbsv(handle, uplo, trans, diag, n, k, A, lda, x, incx):
    """Solves a triangular banded system (double precision real)."""
    assert _libcublas
    status = _libcublas.cublasDtbsv_v2(handle, _CUBLAS_FILL_MODE[uplo], _CUBLAS_OP[trans], _CUBLAS_DIAG[diag], n, k, int(A), lda, int(x), incx)
    cublasCheckStatus(status)


cublasDtbsv.__doc__ = _TBSV_doc.substitute()

_libcublas.cublasCtbsv_v2.restype = int
_libcublas.cublasCtbsv_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]


def cublasCtbsv(handle, uplo, trans, diag, n, k, A, lda, x, incx):
    """Solves a triangular banded system (single precision complex)."""
    assert _libcublas
    status = _libcublas.cublasCtbsv_v2(handle, _CUBLAS_FILL_MODE[uplo], _CUBLAS_OP[trans], _CUBLAS_DIAG[diag], n, k, int(A), lda, int(x), incx)
    cublasCheckStatus(status)


cublasCtbsv.__doc__ = _TBSV_doc.substitute()

_libcublas.cublasZtbsv_v2.restype = int
_libcublas.cublasZtbsv_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]


def cublasZtbsv(handle, uplo, trans, diag, n, k, A, lda, x, incx):
    """Solves a triangular banded system (double precision complex)."""
    assert _libcublas
    status = _libcublas.cublasZtbsv_v2(handle, _CUBLAS_FILL_MODE[uplo], _CUBLAS_OP[trans], _CUBLAS_DIAG[diag], n, k, int(A), lda, int(x), incx)
    cublasCheckStatus(status)


cublasZtbsv.__doc__ = _TBSV_doc.substitute()


# STPMV, DTPMV, CTPMV, ZTPMV
_TPMV_doc = Template(
    """
    Matrix-vector product for a triangular packed matrix.

    Computes the product `A*x`, where `A` is a triangular packed matrix.

    Parameters
    ----------
    handle : int
        CUBLAS context.
    uplo : char
        Specifies whether `A` is upper or lower triangular:
        'U' or 'u': `A` is upper packed.
        'L' or 'l': `A` is lower packed.
    trans : char
        Specifies the form of the matrix `A`:
        'N' or 'n': `A` is not transposed or conjugated.
        'T' or 't': `A` is transposed.
        'C' or 'c': `A` is transposed and conjugated.
    diag : char
        Specifies whether `A` is unit triangular:
        'U' or 'u': `A` is unit triangular.
        'N' or 'n': `A` is not unit triangular.
    n : int
        Number of columns of `A`.
    AP : ctypes.c_void_p
        Pointer to the packed triangular matrix `AP`.
    x : ctypes.c_void_p
        Pointer to the vector `x`.
    incx : int
        Storage spacing between elements of `x`.

    References
    ----------
    `cublas<t>tpmv <http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-tpmv>`_
    """
)

_libcublas.cublasStpmv_v2.restype = int
_libcublas.cublasStpmv_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]


def cublasStpmv(handle, uplo, trans, diag, n, AP, x, incx):
    """Matrix-vector product (single precision real triangular packed)."""
    assert _libcublas
    status = _libcublas.cublasStpmv_v2(handle, _CUBLAS_FILL_MODE[uplo], _CUBLAS_OP[trans], _CUBLAS_DIAG[diag], n, int(AP), int(x), incx)
    cublasCheckStatus(status)


cublasStpmv.__doc__ = _TPMV_doc.substitute()

_libcublas.cublasCtpmv_v2.restype = int
_libcublas.cublasCtpmv_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]


def cublasCtpmv(handle, uplo, trans, diag, n, AP, x, incx):
    """Matrix-vector product (single precision complex triangular packed)."""
    assert _libcublas
    status = _libcublas.cublasCtpmv_v2(handle, _CUBLAS_FILL_MODE[uplo], _CUBLAS_OP[trans], _CUBLAS_DIAG[diag], n, int(AP), int(x), incx)
    cublasCheckStatus(status)


cublasCtpmv.__doc__ = _TPMV_doc.substitute()

_libcublas.cublasDtpmv_v2.restype = int
_libcublas.cublasDtpmv_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]


def cublasDtpmv(handle, uplo, trans, diag, n, AP, x, incx):
    """Matrix-vector product (double precision real triangular packed)."""
    assert _libcublas
    status = _libcublas.cublasDtpmv_v2(handle, _CUBLAS_FILL_MODE[uplo], _CUBLAS_OP[trans], _CUBLAS_DIAG[diag], n, int(AP), int(x), incx)
    cublasCheckStatus(status)


cublasDtpmv.__doc__ = _TPMV_doc.substitute()

_libcublas.cublasZtpmv_v2.restype = int
_libcublas.cublasZtpmv_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]


def cublasZtpmv(handle, uplo, trans, diag, n, AP, x, incx):
    """Matrix-vector product (double precision complex triangular packed)."""
    assert _libcublas
    status = _libcublas.cublasZtpmv_v2(handle, _CUBLAS_FILL_MODE[uplo], _CUBLAS_OP[trans], _CUBLAS_DIAG[diag], n, int(AP), int(x), incx)
    cublasCheckStatus(status)


cublasZtpmv.__doc__ = _TPMV_doc.substitute()


# STPSV, DTPSV, CTPSV, ZTPSV
_TPSV_doc = Template(
    """
    Solves a triangular packed system with one right-hand side.

    Solves the system `A*x = b`, where `A` is a triangular packed matrix.

    Parameters
    ----------
    handle : int
        CUBLAS context.
    uplo : char
        Specifies whether `A` is upper or lower triangular:
        'U' or 'u': `A` is upper packed.
        'L' or 'l': `A` is lower packed.
    trans : char
        Specifies the form of the matrix `A`:
        'N' or 'n': `A` is not transposed or conjugated.
        'T' or 't': `A` is transposed.
        'C' or 'c': `A` is transposed and conjugated.
    diag : char
        Specifies whether `A` is unit triangular:
        'U' or 'u': `A` is unit triangular.
        'N' or 'n': `A` is not unit triangular.
    n : int
        Number of columns of `A`.
    AP : ctypes.c_void_p
        Pointer to the packed triangular matrix `AP`.
    x : ctypes.c_void_p
        Pointer to the vector `x`. The solution is stored here.
    incx : int
        Storage spacing between elements of `x`.

    References
    ----------
    `cublas<t>tpsv <http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-tpsv>`_
    """
)

_libcublas.cublasStpsv_v2.restype = int
_libcublas.cublasStpsv_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]


def cublasStpsv(handle, uplo, trans, diag, n, AP, x, incx):
    """Solves a triangular packed system (single precision real)."""
    assert _libcublas
    status = _libcublas.cublasStpsv_v2(handle, _CUBLAS_FILL_MODE[uplo], _CUBLAS_OP[trans], _CUBLAS_DIAG[diag], n, int(AP), int(x), incx)
    cublasCheckStatus(status)


cublasStpsv.__doc__ = _TPSV_doc.substitute()

_libcublas.cublasDtpsv_v2.restype = int
_libcublas.cublasDtpsv_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]


def cublasDtpsv(handle, uplo, trans, diag, n, AP, x, incx):
    """Solves a triangular packed system (double precision real)."""
    assert _libcublas
    status = _libcublas.cublasDtpsv_v2(handle, _CUBLAS_FILL_MODE[uplo], _CUBLAS_OP[trans], _CUBLAS_DIAG[diag], n, int(AP), int(x), incx)
    cublasCheckStatus(status)


cublasDtpsv.__doc__ = _TPSV_doc.substitute()

_libcublas.cublasCtpsv_v2.restype = int
_libcublas.cublasCtpsv_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]


def cublasCtpsv(handle, uplo, trans, diag, n, AP, x, incx):
    """Solves a triangular packed system (single precision complex)."""
    assert _libcublas
    status = _libcublas.cublasCtpsv_v2(handle, _CUBLAS_FILL_MODE[uplo], _CUBLAS_OP[trans], _CUBLAS_DIAG[diag], n, int(AP), int(x), incx)
    cublasCheckStatus(status)


cublasCtpsv.__doc__ = _TPSV_doc.substitute()

_libcublas.cublasZtpsv_v2.restype = int
_libcublas.cublasZtpsv_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]


def cublasZtpsv(handle, uplo, trans, diag, n, AP, x, incx):
    """Solves a triangular packed system (double precision complex)."""
    assert _libcublas
    status = _libcublas.cublasZtpsv_v2(handle, _CUBLAS_FILL_MODE[uplo], _CUBLAS_OP[trans], _CUBLAS_DIAG[diag], n, int(AP), int(x), incx)
    cublasCheckStatus(status)


cublasZtpsv.__doc__ = _TPSV_doc.substitute()


# STRMV, DTRMV, CTRMV, ZTRMV
_TRMV_doc = Template(
    """
    Matrix-vector product for a triangular matrix.

    Computes the product `A*x`, where `A` is a triangular matrix.

    Parameters
    ----------
    handle : int
        CUBLAS context.
    uplo : char
        Specifies whether `A` is upper or lower triangular:
        'U' or 'u': `A` is upper triangular.
        'L' or 'l': `A` is lower triangular.
    trans : char
        Specifies the form of the matrix `A`:
        'N' or 'n': `A` is not transposed or conjugated.
        'T' or 't': `A` is transposed.
        'C' or 'c': `A` is transposed and conjugated.
    diag : char
        Specifies whether `A` is unit triangular:
        'U' or 'u': `A` is unit triangular.
        'N' or 'n': `A` is not unit triangular.
    n : int
        Number of columns of `A`.
    A : ctypes.c_void_p
        Pointer to the matrix `A`.
    lda : int
        Leading dimension of `A`.
    x : ctypes.c_void_p
        Pointer to the vector `x`.
    inx : int
        Storage spacing between elements of `x`.

    References
    ----------
    `cublas<t>trmv <http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-trmv>`_
    """
)

_libcublas.cublasStrmv_v2.restype = int
_libcublas.cublasStrmv_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]


def cublasStrmv(handle, uplo, trans, diag, n, A, lda, x, inx):
    """Matrix-vector product (single precision real triangular)."""
    assert _libcublas
    status = _libcublas.cublasStrmv_v2(handle, _CUBLAS_FILL_MODE[uplo], _CUBLAS_OP[trans], _CUBLAS_DIAG[diag], n, int(A), lda, int(x), inx)
    cublasCheckStatus(status)


cublasStrmv.__doc__ = _TRMV_doc.substitute()

_libcublas.cublasCtrmv_v2.restype = int
_libcublas.cublasCtrmv_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]


def cublasCtrmv(handle, uplo, trans, diag, n, A, lda, x, incx):
    """Matrix-vector product (single precision complex triangular)."""
    assert _libcublas
    status = _libcublas.cublasCtrmv_v2(handle, _CUBLAS_FILL_MODE[uplo], _CUBLAS_OP[trans], _CUBLAS_DIAG[diag], n, int(A), lda, int(x), incx)
    cublasCheckStatus(status)


cublasCtrmv.__doc__ = _TRMV_doc.substitute()

_libcublas.cublasDtrmv_v2.restype = int
_libcublas.cublasDtrmv_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]


def cublasDtrmv(handle, uplo, trans, diag, n, A, lda, x, inx):
    """Matrix-vector product (double precision real triangular)."""
    assert _libcublas
    status = _libcublas.cublasDtrmv_v2(handle, _CUBLAS_FILL_MODE[uplo], _CUBLAS_OP[trans], _CUBLAS_DIAG[diag], n, int(A), lda, int(x), inx)
    cublasCheckStatus(status)


cublasDtrmv.__doc__ = _TRMV_doc.substitute()

_libcublas.cublasZtrmv_v2.restype = int
_libcublas.cublasZtrmv_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]


def cublasZtrmv(handle, uplo, trans, diag, n, A, lda, x, incx):
    """Matrix-vector product (double precision complex triangular)."""
    assert _libcublas
    status = _libcublas.cublasZtrmv_v2(handle, _CUBLAS_FILL_MODE[uplo], _CUBLAS_OP[trans], _CUBLAS_DIAG[diag], n, int(A), lda, int(x), incx)
    cublasCheckStatus(status)


cublasZtrmv.__doc__ = _TRMV_doc.substitute()


# STRSV, DTRSV, CTRSV, ZTRSV
_TRSV_doc = Template(
    """
    Solves a triangular system with one right-hand side.

    Solves the system `A*x = b`, where `A` is a triangular matrix.

    Parameters
    ----------
    handle : int
        CUBLAS context.
    uplo : char
        Specifies whether `A` is upper or lower triangular:
        'U' or 'u': `A` is upper triangular.
        'L' or 'l': `A` is lower triangular.
    trans : char
        Specifies the form of the matrix `A`:
        'N' or 'n': `A` is not transposed or conjugated.
        'T' or 't': `A` is transposed.
        'C' or 'c': `A` is transposed and conjugated.
    diag : char
        Specifies whether `A` is unit triangular:
        'U' or 'u': `A` is unit triangular.
        'N' or 'n': `A` is not unit triangular.
    n : int
        Number of columns of `A`.
    A : ctypes.c_void_p
        Pointer to the matrix `A`.
    lda : int
        Leading dimension of `A`.
    x : ctypes.c_void_p
        Pointer to the vector `x`. The solution is stored here.
    incx : int
        Storage spacing between elements of `x`.

    References
    ----------
    `cublas<t>trsv <http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-trsv>`_
    """
)

_libcublas.cublasStrsv_v2.restype = int
_libcublas.cublasStrsv_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]


def cublasStrsv(handle, uplo, trans, diag, n, A, lda, x, incx):
    """Solves a triangular system (single precision real)."""
    assert _libcublas
    status = _libcublas.cublasStrsv_v2(handle, _CUBLAS_FILL_MODE[uplo], _CUBLAS_OP[trans], _CUBLAS_DIAG[diag], n, int(A), lda, int(x), incx)
    cublasCheckStatus(status)


cublasStrsv.__doc__ = _TRSV_doc.substitute()

_libcublas.cublasDtrsv_v2.restype = int
_libcublas.cublasDtrsv_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]


def cublasDtrsv(handle, uplo, trans, diag, n, A, lda, x, incx):
    """Solves a triangular system (double precision real)."""
    assert _libcublas
    status = _libcublas.cublasDtrsv_v2(handle, _CUBLAS_FILL_MODE[uplo], _CUBLAS_OP[trans], _CUBLAS_DIAG[diag], n, int(A), lda, int(x), incx)
    cublasCheckStatus(status)


cublasDtrsv.__doc__ = _TRSV_doc.substitute()

_libcublas.cublasCtrsv_v2.restype = int
_libcublas.cublasCtrsv_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]


def cublasCtrsv(handle, uplo, trans, diag, n, A, lda, x, incx):
    """Solves a triangular system (single precision complex)."""
    assert _libcublas
    status = _libcublas.cublasCtrsv_v2(handle, _CUBLAS_FILL_MODE[uplo], _CUBLAS_OP[trans], _CUBLAS_DIAG[diag], n, int(A), lda, int(x), incx)
    cublasCheckStatus(status)


cublasCtrsv.__doc__ = _TRSV_doc.substitute()

_libcublas.cublasZtrsv_v2.restype = int
_libcublas.cublasZtrsv_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]


def cublasZtrsv(handle, uplo, trans, diag, n, A, lda, x, incx):
    """Solves a triangular system (double precision complex)."""
    assert _libcublas
    status = _libcublas.cublasZtrsv_v2(handle, _CUBLAS_FILL_MODE[uplo], _CUBLAS_OP[trans], _CUBLAS_DIAG[diag], n, int(A), lda, int(x), incx)
    cublasCheckStatus(status)


cublasZtrsv.__doc__ = _TRSV_doc.substitute()


# HEMV, ZHEMV
_HEMV_doc = Template(
    """
    Matrix-vector product for a Hermitian matrix.

    Computes the product `alpha*A*x + beta*y`, where `A` is a Hermitian matrix.

    Parameters
    ----------
    handle : int
        CUBLAS context.
    uplo : char
        Specifies whether `A` is upper or lower triangular:
        'U' or 'u': `A` is upper triangular.
        'L' or 'l': `A` is lower triangular.
    n : int
        Number of columns of `A`.
    alpha : ${a_type}
        Scalar multiplier for matrix `A`.
    A : ctypes.c_void_p
        Pointer to the matrix `A`.
    lda : int
        Leading dimension of `A`.
    x : ctypes.c_void_p
        Pointer to the vector `x`.
    incx : int
        Storage spacing between elements of `x`.
    beta : ${a_type}
        Scalar multiplier for vector `y`.
    y : ctypes.c_void_p
        Pointer to the vector `y`. The result is stored here.
    incy : int
        Storage spacing between elements of `y`.

    References
    ----------
    `cublas<t>hemv <http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-hemv>`_
    """
)

_libcublas.cublasChemv_v2.restype = int
_libcublas.cublasChemv_v2.argtypes = [
    _types.handle,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
]


def cublasChemv(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy):
    """Matrix-vector product (single precision complex Hermitian)."""
    assert _libcublas
    status = _libcublas.cublasChemv_v2(
        handle, _CUBLAS_FILL_MODE[uplo], n, ctypes.byref(cuda.cuFloatComplex(alpha.real, alpha.imag)), int(A), lda, int(x), incx, ctypes.byref(cuda.cuFloatComplex(beta.real, beta.imag)), int(y), incy
    )
    cublasCheckStatus(status)


cublasChemv.__doc__ = _HEMV_doc.substitute(a_type="numpy.complex64")

_libcublas.cublasZhemv_v2.restype = int
_libcublas.cublasZhemv_v2.argtypes = [
    _types.handle,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
]


def cublasZhemv(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy):
    """Matrix-vector product (double precision complex Hermitian)."""
    assert _libcublas
    status = _libcublas.cublasZhemv_v2(
        handle,
        _CUBLAS_FILL_MODE[uplo],
        n,
        ctypes.byref(cuda.cuDoubleComplex(alpha.real, alpha.imag)),
        int(A),
        lda,
        int(x),
        incx,
        ctypes.byref(cuda.cuDoubleComplex(beta.real, beta.imag)),
        int(y),
        incy,
    )
    cublasCheckStatus(status)


cublasZhemv.__doc__ = _HEMV_doc.substitute(a_type="numpy.complex128")


# CHBMV, ZHBMV
_HBMV_doc = Template(
    """
    Matrix-vector product for a Hermitian banded matrix.

    Computes the product `alpha*A*x + beta*y`, where `A` is a Hermitian banded
    matrix.

    Parameters
    ----------
    handle : int
        CUBLAS context.
    uplo : char
        Specifies whether `A` is upper or lower triangular:
        'U' or 'u': `A` is upper banded.
        'L' or 'l': `A` is lower banded.
    n : int
        Number of columns of `A`.
    k : int
        Number of super- or sub-diagonals of `A`.
    alpha : ${a_type}
        Scalar multiplier for matrix `A`.
    A : ctypes.c_void_p
        Pointer to the matrix `A`.
    lda : int
        Leading dimension of `A`.
    x : ctypes.c_void_p
        Pointer to the vector `x`.
    incx : int
        Storage spacing between elements of `x`.
    beta : ${a_type}
        Scalar multiplier for vector `y`.
    y : ctypes.c_void_p
        Pointer to the vector `y`. The result is stored here.
    incy : int
        Storage spacing between elements of `y`.

    References
    ----------
    `cublas<t>hbmv <http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-hbmv>`_
    """
)

_libcublas.cublasChbmv_v2.restype = int
_libcublas.cublasChbmv_v2.argtypes = [
    _types.handle,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
]


def cublasChbmv(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy):
    """Matrix-vector product (single precision complex Hermitian banded)."""
    assert _libcublas
    status = _libcublas.cublasChbmv_v2(
        handle,
        _CUBLAS_FILL_MODE[uplo],
        n,
        k,
        ctypes.byref(cuda.cuFloatComplex(alpha.real, alpha.imag)),
        int(A),
        lda,
        int(x),
        incx,
        ctypes.byref(cuda.cuFloatComplex(beta.real, beta.imag)),
        int(y),
        incy,
    )
    cublasCheckStatus(status)


cublasChbmv.__doc__ = _HBMV_doc.substitute(a_type="numpy.complex64")

_libcublas.cublasZhbmv_v2.restype = int
_libcublas.cublasZhbmv_v2.argtypes = [
    _types.handle,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
]


def cublasZhbmv(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy):
    """Matrix-vector product (double precision complex Hermitian banded)."""
    assert _libcublas
    status = _libcublas.cublasZhbmv_v2(
        handle,
        _CUBLAS_FILL_MODE[uplo],
        n,
        k,
        ctypes.byref(cuda.cuDoubleComplex(alpha.real, alpha.imag)),
        int(A),
        lda,
        int(x),
        incx,
        ctypes.byref(cuda.cuDoubleComplex(beta.real, beta.imag)),
        int(y),
        incy,
    )
    cublasCheckStatus(status)


cublasZhbmv.__doc__ = _HBMV_doc.substitute(a_type="numpy.complex128")


# CHPMV, ZHPMV
_HPMV_doc = Template(
    """
    Matrix-vector product for a Hermitian packed matrix.

    Computes the product `alpha*AP*x + beta*y`, where `AP` is a Hermitian packed
    matrix.

    Parameters
    ----------
    handle : int
        CUBLAS context.
    uplo : char
        Specifies whether `AP` is upper or lower triangular:
        'U' or 'u': `AP` is upper packed.
        'L' or 'l': `AP` is lower packed.
    n : int
        Number of elements of the matrix `AP`.
    alpha : ${a_type}
        Scalar multiplier for matrix `AP`.
    AP : ctypes.c_void_p
        Pointer to the packed Hermitian matrix `AP`.
    x : ctypes.c_void_p
        Pointer to the vector `x`.
    incx : int
        Storage spacing between elements of `x`.
    beta : ${a_type}
        Scalar multiplier for vector `y`.
    y : ctypes.c_void_p
        Pointer to the vector `y`. The result is stored here.
    incy : int
        Storage spacing between elements of `y`.

    References
    ----------
    `cublas<t>hpmv <http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-tpmv>`_
    """
)

_libcublas.cublasChpmv_v2.restype = int
_libcublas.cublasChpmv_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]


def cublasChpmv(handle, uplo, n, alpha, AP, x, incx, beta, y, incy):
    """Matrix-vector product (single precision complex Hermitian packed)."""
    assert _libcublas
    status = _libcublas.cublasChpmv_v2(
        handle, _CUBLAS_FILL_MODE[uplo], n, ctypes.byref(cuda.cuFloatComplex(alpha.real, alpha.imag)), int(AP), int(x), incx, ctypes.byref(cuda.cuFloatComplex(beta.real, beta.imag)), int(y), incy
    )
    cublasCheckStatus(status)


cublasChpmv.__doc__ = _HPMV_doc.substitute(a_type="numpy.complex64")

_libcublas.cublasZhpmv_v2.restype = int
_libcublas.cublasZhpmv_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]


def cublasZhpmv(handle, uplo, n, alpha, AP, x, incx, beta, y, incy):
    """Matrix-vector product (double precision complex Hermitian packed)."""
    assert _libcublas
    status = _libcublas.cublasZhpmv_v2(
        handle, _CUBLAS_FILL_MODE[uplo], n, ctypes.byref(cuda.cuDoubleComplex(alpha.real, alpha.imag)), int(AP), int(x), incx, ctypes.byref(cuda.cuDoubleComplex(beta.real, beta.imag)), int(y), incy
    )
    cublasCheckStatus(status)


cublasZhpmv.__doc__ = _HPMV_doc.substitute(a_type="numpy.complex128")


# CHER, ZHER
_HER_doc = Template(
    """
    Rank-1 operation on a Hermitian matrix.

    Updates the Hermitian matrix `A` with a rank-1 operation:
    `A = alpha*x*x^H + A`.

    Parameters
    ----------
    handle : int
        CUBLAS context.
    uplo : char
        Specifies whether `A` is upper or lower triangular:
        'U' or 'u': `A` is upper triangular.
        'L' or 'l': `A` is lower triangular.
    n : int
        Number of columns of `A`.
    alpha : ${a_type}
        Scalar multiplier.
    x : ctypes.c_void_p
        Pointer to the vector `x`.
    incx : int
        Storage spacing between elements of `x`.
    A : ctypes.c_void_p
        Pointer to the matrix `A`.
    lda : int
        Leading dimension of `A`.

    References
    ----------
    `cublas<t>her <http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-her>`_
    """
)

_libcublas.cublasCher_v2.restype = int
_libcublas.cublasCher_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]


def cublasCher(handle, uplo, n, alpha, x, incx, A, lda):
    """Rank-1 operation (single precision complex Hermitian)."""
    assert _libcublas
    status = _libcublas.cublasCher_v2(handle, _CUBLAS_FILL_MODE[uplo], n, alpha, int(x), incx, int(A), lda)
    cublasCheckStatus(status)


cublasCher.__doc__ = _HER_doc.substitute(a_type="numpy.float32")

_libcublas.cublasZher_v2.restype = int
_libcublas.cublasZher_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]


def cublasZher(handle, uplo, n, alpha, x, incx, A, lda):
    """Rank-1 operation (double precision complex Hermitian)."""
    assert _libcublas
    status = _libcublas.cublasZher_v2(handle, _CUBLAS_FILL_MODE[uplo], n, alpha, int(x), incx, int(A), lda)
    cublasCheckStatus(status)


cublasZher.__doc__ = _HER_doc.substitute(a_type="numpy.float64")


# CHER2, ZHER2
_HER2_doc = Template(
    """
    Rank-2 operation on a Hermitian matrix.

    Updates the Hermitian matrix `A` with a rank-2 operation:
    `A = alpha*x*y^H + conj(alpha)*y*x^H + A`.

    Parameters
    ----------
    handle : int
        CUBLAS context.
    uplo : char
        Specifies whether `A` is upper or lower triangular:
        'U' or 'u': `A` is upper triangular.
        'L' or 'l': `A` is lower triangular.
    n : int
        Number of columns of `A`.
    alpha : ${a_type}
        Scalar multiplier.
    x : ctypes.c_void_p
        Pointer to the first vector `x`.
    incx : int
        Storage spacing between elements of `x`.
    y : ctypes.c_void_p
        Pointer to the second vector `y`.
    incy : int
        Storage spacing between elements of `y`.
    A : ctypes.c_void_p
        Pointer to the matrix `A`.
    lda : int
        Leading dimension of `A`.

    References
    ----------
    `cublas<t>her2 <http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-her2>`_
    """
)

_libcublas.cublasCher2_v2.restype = int
_libcublas.cublasCher2_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]


def cublasCher2(handle, uplo, n, alpha, x, incx, y, incy, A, lda):
    """Rank-2 operation (single precision complex Hermitian)."""
    assert _libcublas
    status = _libcublas.cublasCher2_v2(handle, _CUBLAS_FILL_MODE[uplo], n, ctypes.byref(cuda.cuFloatComplex(alpha.real, alpha.imag)), int(x), incx, int(y), incy, int(A), lda)
    cublasCheckStatus(status)


cublasCher2.__doc__ = _HER2_doc.substitute(a_type="numpy.complex64")

_libcublas.cublasZher2_v2.restype = int
_libcublas.cublasZher2_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]


def cublasZher2(handle, uplo, n, alpha, x, incx, y, incy, A, lda):
    """Rank-2 operation (double precision complex Hermitian)."""
    assert _libcublas
    status = _libcublas.cublasZher2_v2(handle, _CUBLAS_FILL_MODE[uplo], n, ctypes.byref(cuda.cuDoubleComplex(alpha.real, alpha.imag)), int(x), incx, int(y), incy, int(A), lda)
    cublasCheckStatus(status)


cublasZher2.__doc__ = _HER2_doc.substitute(a_type="numpy.complex128")


# CHPR, ZHPR
_HPR_doc = Template(
    """
    Rank-1 operation on a Hermitian packed matrix.

    Updates the Hermitian packed matrix `AP` with a rank-1 operation:
    `AP = alpha*x*x^H + AP`.

    Parameters
    ----------
    handle : int
        CUBLAS context.
    uplo : char
        Specifies whether `AP` is upper or lower triangular:
        'U' or 'u': `AP` is upper packed.
        'L' or 'l': `AP` is lower packed.
    n : int
        Number of elements of the matrix `AP`.
    alpha : ${a_type}
        Scalar multiplier.
    x : ctypes.c_void_p
        Pointer to the vector `x`.
    incx : int
        Storage spacing between elements of `x`.
    AP : ctypes.c_void_p
        Pointer to the packed Hermitian matrix `AP`.

    References
    ----------
    `cublas<t>hpr <http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-hpr>`_
    """
)

_libcublas.cublasChpr_v2.restype = int
_libcublas.cublasChpr_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]


def cublasChpr(handle, uplo, n, alpha, x, incx, AP):
    """Rank-1 operation (single precision complex Hermitian packed)."""
    assert _libcublas
    status = _libcublas.cublasChpr_v2(handle, _CUBLAS_FILL_MODE[uplo], n, ctypes.byref(ctypes.c_float(alpha)), int(x), incx, int(AP))
    cublasCheckStatus(status)


cublasChpr.__doc__ = _HPR_doc.substitute(a_type="numpy.float32")

_libcublas.cublasZhpr_v2.restype = int
_libcublas.cublasZhpr_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]


def cublasZhpr(handle, uplo, n, alpha, x, incx, AP):
    """Rank-1 operation (double precision complex Hermitian packed)."""
    assert _libcublas
    status = _libcublas.cublasZhpr_v2(handle, _CUBLAS_FILL_MODE[uplo], n, ctypes.byref(ctypes.c_double(alpha)), int(x), incx, int(AP))
    cublasCheckStatus(status)


cublasZhpr.__doc__ = _HPR_doc.substitute(a_type="numpy.float64")


# CHPR2, ZHPR2
_HPR2_doc = Template(
    """
    Rank-2 operation on a Hermitian packed matrix.

    Updates the Hermitian packed matrix `AP` with a rank-2 operation:
    `AP = alpha*x*y^H + conj(alpha)*y*x^H + AP`.

    Parameters
    ----------
    handle : int
        CUBLAS context.
    uplo : char
        Specifies whether `AP` is upper or lower triangular:
        'U' or 'u': `AP` is upper packed.
        'L' or 'l': `AP` is lower packed.
    n : int
        Number of elements of the matrix `AP`.
    alpha : ${a_type}
        Scalar multiplier.
    x : ctypes.c_void_p
        Pointer to the first vector `x`.
    incx : int
        Storage spacing between elements of `x`.
    y : ctypes.c_void_p
        Pointer to the second vector `y`.
    incy : int
        Storage spacing between elements of `y`.
    AP : ctypes.c_void_p
        Pointer to the packed Hermitian matrix `AP`.

    References
    ----------
    `cublas<t>hpr2 <http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-hpr2>`_
    """
)

_libcublas.cublasChpr2.restype = int
_libcublas.cublasChpr2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]


def cublasChpr2(handle, uplo, n, alpha, x, incx, y, incy, AP):
    """Rank-2 operation (single precision complex Hermitian packed)."""
    assert _libcublas
    status = _libcublas.cublasChpr2_v2(handle, _CUBLAS_FILL_MODE[uplo], n, ctypes.byref(cuda.cuFloatComplex(alpha.real, alpha.imag)), int(x), incx, int(y), incy, int(AP))
    cublasCheckStatus(status)


cublasChpr2.__doc__ = _HPR2_doc.substitute(a_type="numpy.complex64")

_libcublas.cublasZhpr2_v2.restype = int
_libcublas.cublasZhpr2_v2.argtypes = [_types.handle, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]


def cublasZhpr2(handle, uplo, n, alpha, x, incx, y, incy, AP):
    """Rank-2 operation (double precision complex Hermitian packed)."""
    assert _libcublas
    status = _libcublas.cublasZhpr2_v2(handle, _CUBLAS_FILL_MODE[uplo], n, ctypes.byref(cuda.cuDoubleComplex(alpha.real, alpha.imag)), int(x), incx, int(y), incy, int(AP))
    cublasCheckStatus(status)


cublasZhpr2.__doc__ = _HPR2_doc.substitute(a_type="numpy.complex128")


# SGEMM, CGEMM, DGEMM, ZGEMM
_GEMM_doc = Template(
    """
    Matrix-matrix product.

    Computes the product `alpha*op(A)*B + beta*C`, where `op(A)` is `A` or `A^T`
    or `A^H`.

    Parameters
    ----------
    handle : int
        CUBLAS context.
    transa : char
        Specifies the form of the matrix `A`:
        'N' or 'n': `A` is not transposed or conjugated.
        'T' or 't': `A` is transposed.
        'C' or 'c': `A` is transposed and conjugated.
    transb : char
        Specifies the form of the matrix `B`:
        'N' or 'n': `B` is not transposed or conjugated.
        'T' or 't': `B` is transposed.
        'C' or 'c': `B` is transposed and conjugated.
    m : int
        Number of rows of matrix `A` and `C`.
    n : int
        Number of columns of matrix `B` and `C`.
    k : int
        Number of columns of matrix `A` and rows of matrix `B`.
    alpha : ${a_type}
        Scalar multiplier for `op(A)*B`.
    A : ctypes.c_void_p
        Pointer to the matrix `A`.
    lda : int
        Leading dimension of `A`.
    B : ctypes.c_void_p
        Pointer to the matrix `B`.
    ldb : int
        Leading dimension of `B`.
    beta : ${a_type}
        Scalar multiplier for matrix `C`.
    C : ctypes.c_void_p
        Pointer to the matrix `C`. The result is stored here.
    ldc : int
        Leading dimension of `C`.

    References
    ----------
    `cublas<t>gemm <http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm>`_
    """
)

_libcublas.cublasSgemm_v2.restype = int
_libcublas.cublasSgemm_v2.argtypes = [
    _types.handle,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
]


def cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
    """Matrix-matrix product (single precision real)."""
    assert _libcublas
    status = _libcublas.cublasSgemm_v2(
        handle, _CUBLAS_OP[transa], _CUBLAS_OP[transb], m, n, k, ctypes.byref(ctypes.c_float(alpha)), int(A), lda, int(B), ldb, ctypes.byref(ctypes.c_float(beta)), int(C), ldc
    )
    cublasCheckStatus(status)


cublasSgemm.__doc__ = _GEMM_doc.substitute(a_type="numpy.float32")

_libcublas.cublasCgemm_v2.restype = int
_libcublas.cublasCgemm_v2.argtypes = [
    _types.handle,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
]


def cublasCgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
    """Matrix-matrix product (single precision complex)."""
    assert _libcublas
    status = _libcublas.cublasCgemm_v2(
        handle,
        _CUBLAS_OP[transa],
        _CUBLAS_OP[transb],
        m,
        n,
        k,
        ctypes.byref(cuda.cuFloatComplex(alpha.real, alpha.imag)),
        int(A),
        lda,
        int(B),
        ldb,
        ctypes.byref(cuda.cuFloatComplex(beta.real, beta.imag)),
        int(C),
        ldc,
    )
    cublasCheckStatus(status)


cublasCgemm.__doc__ = _GEMM_doc.substitute(a_type="numpy.complex64")

_libcublas.cublasDgemm_v2.restype = int
_libcublas.cublasDgemm_v2.argtypes = [
    _types.handle,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
]


def cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
    """Matrix-matrix product (double precision real)."""
    assert _libcublas
    status = _libcublas.cublasDgemm_v2(
        handle, _CUBLAS_OP[transa], _CUBLAS_OP[transb], m, n, k, ctypes.byref(ctypes.c_double(alpha)), int(A), lda, int(B), ldb, ctypes.byref(ctypes.c_double(beta)), int(C), ldc
    )
    cublasCheckStatus(status)


cublasDgemm.__doc__ = _GEMM_doc.substitute(a_type="numpy.float64")

_libcublas.cublasZgemm_v2.restype = int
_libcublas.cublasZgemm_v2.argtypes = [
    _types.handle,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
]


def cublasZgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
    """Matrix-matrix product (double precision complex)."""
    assert _libcublas
    status = _libcublas.cublasZgemm_v2(
        handle,
        _CUBLAS_OP[transa],
        _CUBLAS_OP[transb],
        m,
        n,
        k,
        ctypes.byref(cuda.cuDoubleComplex(alpha.real, alpha.imag)),
        int(A),
        lda,
        int(B),
        ldb,
        ctypes.byref(cuda.cuDoubleComplex(beta.real, beta.imag)),
        int(C),
        ldc,
    )
    cublasCheckStatus(status)


cublasZgemm.__doc__ = _GEMM_doc.substitute(a_type="numpy.complex128")


# SSYMM, DSYMM, CSYMM, ZSYMM
_SYMM_doc = Template(
    """
    Matrix-matrix product for a symmetric matrix.

    Computes the product `alpha*op(A)*B + beta*C`, where `A` is a symmetric matrix.

    Parameters
    ----------
    handle : int
        CUBLAS context.
    side : char
        Specifies whether `A` is on the left or right:
        'L' or 'l': `A` is on the left.
        'R' or 'r': `A` is on the right.
    uplo : char
        Specifies whether `A` is upper or lower triangular:
        'U' or 'u': `A` is upper triangular.
        'L' or 'l': `A` is lower triangular.
    m : int
        Number of rows of `A` and `C` if `side` is 'L' or 'l'.
        Number of rows of `B` if `side` is 'R' or 'r'.
    n : int
        Number of columns of `B` and `C` if `side` is 'L' or 'l'.
        Number of columns of `A` if `side` is 'R' or 'r'.
    alpha : ${a_type}
        Scalar multiplier for matrix `A`.
    A : ctypes.c_void_p
        Pointer to the matrix `A`.
    lda : int
        Leading dimension of `A`.
    B : ctypes.c_void_p
        Pointer to the matrix `B`.
    ldb : int
        Leading dimension of `B`.
    beta : ${a_type}
        Scalar multiplier for matrix `C`.
    C : ctypes.c_void_p
        Pointer to the matrix `C`. The result is stored here.
    ldc : int
        Leading dimension of `C`.

    References
    ----------
    `cublas<t>symm <http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-symm>`_
    """
)

_libcublas.cublasSsymm_v2.restype = int
_libcublas.cublasSsymm_v2.argtypes = [
    _types.handle,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
]


def cublasSsymm(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc):
    """Matrix-matrix product (single precision real symmetric)."""
    assert _libcublas
    status = _libcublas.cublasSsymm_v2(
        handle, _CUBLAS_SIDE_MODE[side], _CUBLAS_FILL_MODE[uplo], m, n, ctypes.byref(ctypes.c_float(alpha)), int(A), lda, ctypes.byref(ctypes.c_float(beta)), int(B), ldb, int(C), ldc
    )
    cublasCheckStatus(status)


cublasSsymm.__doc__ = _SYMM_doc.substitute(a_type="numpy.float32")

_libcublas.cublasDsymm_v2.restype = int
_libcublas.cublasDsymm_v2.argtypes = [
    _types.handle,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
]


def cublasDsymm(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc):
    """Matrix-matrix product (double precision real symmetric)."""
    assert _libcublas
    status = _libcublas.cublasDsymm_v2(
        handle, _CUBLAS_SIDE_MODE[side], _CUBLAS_FILL_MODE[uplo], m, n, ctypes.byref(ctypes.c_double(alpha)), int(A), lda, ctypes.byref(ctypes.c_double(beta)), int(B), ldb, int(C), ldc
    )
    cublasCheckStatus(status)


cublasDsymm.__doc__ = _SYMM_doc.substitute(a_type="numpy.float64")

_libcublas.cublasCsymm_v2.restype = int
_libcublas.cublasCsymm_v2.argtypes = [
    _types.handle,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
]


def cublasCsymm(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc):
    """Matrix-matrix product (single precision complex symmetric)."""
    assert _libcublas
    status = _libcublas.cublasCsymm_v2(
        handle,
        _CUBLAS_SIDE_MODE[side],
        _CUBLAS_FILL_MODE[uplo],
        m,
        n,
        ctypes.byref(cuda.cuFloatComplex(alpha.real, alpha.imag)),
        int(A),
        lda,
        ctypes.byref(cuda.cuFloatComplex(beta.real, beta.imag)),
        int(B),
        ldb,
        int(C),
        ldc,
    )
    cublasCheckStatus(status)


cublasCsymm.__doc__ = _SYMM_doc.substitute(a_type="numpy.complex64")

_libcublas.cublasZsymm_v2.restype = int
_libcublas.cublasZsymm_v2.argtypes = [
    _types.handle,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
]


def cublasZsymm(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc):
    """Matrix-matrix product (double precision complex symmetric)."""
    assert _libcublas
    status = _libcublas.cublasZsymm_v2(
        handle,
        _CUBLAS_SIDE_MODE[side],
        _CUBLAS_FILL_MODE[uplo],
        m,
        n,
        ctypes.byref(cuda.cuDoubleComplex(alpha.real, alpha.imag)),
        int(A),
        lda,
        ctypes.byref(cuda.cuDoubleComplex(beta.real, beta.imag)),
        int(B),
        ldb,
        int(C),
        ldc,
    )
    cublasCheckStatus(status)


cublasZsymm.__doc__ = _SYMM_doc.substitute(a_type="numpy.complex128")


# SSYRK, DSYRK, CSYRK, ZSYRK
_SYRK_doc = Template(
    """
    Rank-k operation on a symmetric matrix.

    Updates the symmetric matrix `C` with a rank-k operation:
    `C = alpha*op(A)*A^T + beta*C` (for real) or
    `C = alpha*op(A)*A^H + beta*C` (for complex).

    Parameters
    ----------
    handle : int
        CUBLAS context.
    uplo : char
        Specifies whether `A` and `C` are upper or lower triangular:
        'U' or 'u': `A` and `C` are upper triangular.
        'L' or 'l': `A` and `C` are lower triangular.
    trans : char
        Specifies the form of the matrix `A`:
        'N' or 'n': `A` is not transposed or conjugated.
        'T' or 't': `A` is transposed.
        'C' or 'c': `A` is transposed and conjugated.
    n : int
        Number of columns of `C`.
    k : int
        Inner dimension of the matrix product.
    alpha : ${a_type}
        Scalar multiplier for `op(A)*A^T` or `op(A)*A^H`.
    A : ctypes.c_void_p
        Pointer to the matrix `A`.
    lda : int
        Leading dimension of `A`.
    beta : ${a_type}
        Scalar multiplier for matrix `C`.
    C : ctypes.c_void_p
        Pointer to the matrix `C`. The result is stored here.
    ldc : int
        Leading dimension of `C`.

    References
    ----------
    `cublas<t>syrk <http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-syrk>`_
    """
)

_libcublas.cublasSsyrk_v2.restype = int
_libcublas.cublasSsyrk_v2.argtypes = [
    _types.handle,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
]


def cublasSsyrk(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc):
    """Rank-k operation (single precision real symmetric)."""
    assert _libcublas
    status = _libcublas.cublasSsyrk_v2(handle, _CUBLAS_FILL_MODE[uplo], _CUBLAS_OP[trans], n, k, ctypes.byref(ctypes.c_float(alpha)), int(A), lda, ctypes.byref(ctypes.c_float(beta)), int(C), ldc)
    cublasCheckStatus(status)


cublasSsyrk.__doc__ = _SYRK_doc.substitute(a_type="numpy.float32")

_libcublas.cublasDsyrk_v2.restype = int
_libcublas.cublasDsyrk_v2.argtypes = [
    _types.handle,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
]


def cublasDsyrk(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc):
    """Rank-k operation (double precision real symmetric)."""
    assert _libcublas
    status = _libcublas.cublasDsyrk_v2(handle, _CUBLAS_FILL_MODE[uplo], _CUBLAS_OP[trans], n, k, ctypes.byref(ctypes.c_double(alpha)), int(A), lda, ctypes.byref(ctypes.c_double(beta)), int(C), ldc)
    cublasCheckStatus(status)


cublasDsyrk.__doc__ = _SYRK_doc.substitute(a_type="numpy.float64")

_libcublas.cublasCsyrk_v2.restype = int
_libcublas.cublasCsyrk_v2.argtypes = [
    _types.handle,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
]


def cublasCsyrk(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc):
    """Rank-k operation (single precision complex symmetric)."""
    assert _libcublas
    status = _libcublas.cublasCsyrk_v2(
        handle,
        _CUBLAS_FILL_MODE[uplo],
        _CUBLAS_OP[trans],
        n,
        k,
        ctypes.byref(cuda.cuFloatComplex(alpha.real, alpha.imag)),
        int(A),
        lda,
        ctypes.byref(cuda.cuFloatComplex(beta.real, beta.imag)),
        int(C),
        ldc,
    )
    cublasCheckStatus(status)


cublasCsyrk.__doc__ = _SYRK_doc.substitute(a_type="numpy.complex64")

_libcublas.cublasZsyrk_v2.restype = int
_libcublas.cublasZsyrk_v2.argtypes = [
    _types.handle,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
]


def cublasZsyrk(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc):
    """Rank-k operation (double precision complex symmetric)."""
    assert _libcublas
    status = _libcublas.cublasZsyrk_v2(
        handle,
        _CUBLAS_FILL_MODE[uplo],
        _CUBLAS_OP[trans],
        n,
        k,
        ctypes.byref(cuda.cuDoubleComplex(alpha.real, alpha.imag)),
        int(A),
        lda,
        ctypes.byref(cuda.cuDoubleComplex(beta.real, beta.imag)),
        int(C),
        ldc,
    )
    cublasCheckStatus(status)


cublasZsyrk.__doc__ = _SYRK_doc.substitute(a_type="numpy.complex128")


# SSYR2K, DSYR2K, CSYR2K, ZSYR2K
_SYR2K_doc = Template(
    """
    Rank-2k operation on a symmetric matrix.

    Updates the symmetric matrix `C` with a rank-2k operation:
    `C = alpha*op(A)*B^T + conj(alpha)*op(B)*A^T + beta*C` (for complex) or
    `C = alpha*op(A)*B^T + alpha*op(B)*A^T + beta*C` (for real).

    Parameters
    ----------
    handle : int
        CUBLAS context.
    uplo : char
        Specifies whether `A`, `B`, and `C` are upper or lower triangular:
        'U' or 'u': `A`, `B`, and `C` are upper triangular.
        'L' or 'l': `A`, `B`, and `C` are lower triangular.
    trans : char
        Specifies the form of the matrices `A` and `B`:
        'N' or 'n': `A` and `B` are not transposed or conjugated.
        'T' or 't': `A` and `B` are transposed.
        'C' or 'c': `A` and `B` are transposed and conjugated.
    n : int
        Number of columns of `C`.
    k : int
        Inner dimension of the matrix products.
    alpha : ${a_type}
        Scalar multiplier for `op(A)*B^T` and `op(B)*A^T`.
    A : ctypes.c_void_p
        Pointer to the matrix `A`.
    lda : int
        Leading dimension of `A`.
    B : ctypes.c_void_p
        Pointer to the matrix `B`.
    ldb : int
        Leading dimension of `B`.
    beta : ${a_type}
        Scalar multiplier for matrix `C`.
    C : ctypes.c_void_p
        Pointer to the matrix `C`. The result is stored here.
    ldc : int
        Leading dimension of `C`.

    References
    ----------
    `cublas<t>syr2k <http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-syr2k>`_
    """
)

_libcublas.cublasSsyr2k_v2.restype = int
_libcublas.cublasSsyr2k_v2.argtypes = [
    _types.handle,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
]


def cublasSsyr2k(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
    """Rank-2k operation (single precision real symmetric)."""
    assert _libcublas
    status = _libcublas.cublasSsyr2k_v2(
        handle, _CUBLAS_FILL_MODE[uplo], _CUBLAS_OP[trans], n, k, ctypes.byref(ctypes.c_float(alpha)), int(A), lda, int(B), ldb, ctypes.byref(ctypes.c_float(beta)), int(C), ldc
    )
    cublasCheckStatus(status)


cublasSsyr2k.__doc__ = _SYR2K_doc.substitute(a_type="numpy.float32")

_libcublas.cublasDsyr2k_v2.restype = int
_libcublas.cublasDsyr2k_v2.argtypes = [
    _types.handle,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
]


def cublasDsyr2k(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
    """Rank-2k operation (double precision real symmetric)."""
    assert _libcublas
    status = _libcublas.cublasDsyr2k_v2(
        handle, _CUBLAS_FILL_MODE[uplo], _CUBLAS_OP[trans], n, k, ctypes.byref(ctypes.c_double(alpha)), int(A), lda, int(B), ldb, ctypes.byref(ctypes.c_double(beta)), int(C), ldc
    )
    cublasCheckStatus(status)


cublasDsyr2k.__doc__ = _SYR2K_doc.substitute(a_type="numpy.float64")

_libcublas.cublasCsyr2k_v2.restype = int
_libcublas.cublasCsyr2k_v2.argtypes = [
    _types.handle,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
]


def cublasCsyr2k(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
    """Rank-2k operation on complex single precision symmetric matrix."""
    assert _libcublas
    status = _libcublas.cublasCsyr2k_v2(
        handle,
        _CUBLAS_FILL_MODE[uplo],
        _CUBLAS_OP[trans],
        n,
        k,
        ctypes.byref(cuda.cuFloatComplex(alpha.real, alpha.imag)),
        int(A),
        lda,
        int(B),
        ldb,
        ctypes.byref(cuda.cuFloatComplex(beta.real, beta.imag)),
        int(C),
        ldc,
    )
    cublasCheckStatus(status)


cublasCsyr2k.__doc__ = _SYR2K_doc.substitute(a_type="numpy.complex64")


_libcublas.cublasZsyr2k_v2.restype = int
_libcublas.cublasZsyr2k_v2.argtypes = [
    _types.handle,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
]


def cublasZsyr2k(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
    """Rank-2k operation on complex double precision symmetric matrix."""
    assert _libcublas
    status = _libcublas.cublasZsyr2k_v2(
        handle,
        _CUBLAS_FILL_MODE[uplo],
        _CUBLAS_OP[trans],
        n,
        k,
        ctypes.byref(cuda.cuDoubleComplex(alpha.real, alpha.imag)),
        int(A),
        lda,
        int(B),
        ldb,
        ctypes.byref(cuda.cuDoubleComplex(beta.real, beta.imag)),
        int(C),
        ldc,
    )
    cublasCheckStatus(status)


cublasZsyr2k.__doc__ = _SYR2K_doc.substitute(a_type="numpy.complex128")


# STRMM, DTRMM, CTRMM, ZTRMM
_TRMM_doc = Template(
    """
    Matrix-matrix multiplication with a triangular matrix.

    Computes:
    `C = alpha*op(A)*B` if `side == 'L'`, or
    `C = alpha*B*op(A)` if `side == 'R'`.

    `A` is a triangular matrix and `B` and `C` are general matrices.

    Parameters
    ----------
    handle : int
        CUBLAS context.
    side : char
        Specifies whether the triangular matrix multiplies from the left
        or right:
        'L' or 'l': left side multiplication.
        'R' or 'r': right side multiplication.
    uplo : char
        Specifies whether `A` is upper or lower triangular:
        'U' or 'u': upper triangular.
        'L' or 'l': lower triangular.
    trans : char
        Specifies the form of `A`:
        'N' or 'n': no transpose.
        'T' or 't': transpose.
        'C' or 'c': conjugate transpose.
    diag : char
        Specifies whether `A` is unit triangular:
        'U' or 'u': unit triangular.
        'N' or 'n': non-unit triangular.
    m : int
        Number of rows of `B` and `C`.
    n : int
        Number of columns of `B` and `C`.
    alpha : ${a_type}
        Scalar multiplier.
    A : ctypes.c_void_p
        Pointer to triangular matrix `A`.
    lda : int
        Leading dimension of `A`.
    B : ctypes.c_void_p
        Pointer to matrix `B`.
    ldb : int
        Leading dimension of `B`.
    C : ctypes.c_void_p
        Pointer to matrix `C`. The result is stored here.
    ldc : int
        Leading dimension of `C`.

    References
    ----------
    `cublas<t>trmm <http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-trmm>`_
    """
)

_libcublas.cublasStrmm_v2.restype = int
_libcublas.cublasStrmm_v2.argtypes = [
    _types.handle,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
]


def cublasStrmm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc):
    """Matrix-matrix product for real single precision triangular matrix"""
    assert _libcublas
    status = _libcublas.cublasStrmm_v2(
        handle, _CUBLAS_SIDE_MODE[side], _CUBLAS_FILL_MODE[uplo], _CUBLAS_OP[trans], _CUBLAS_DIAG[diag], m, n, ctypes.byref(ctypes.c_float(alpha)), int(A), lda, int(B), ldb, int(C), ldc
    )
    cublasCheckStatus(status)

cublasStrmm.__doc__ = _TRMM_doc.substitute(a_type="numpy.float32")

_libcublas.cublasDtrmm_v2.restype = int
_libcublas.cublasDtrmm_v2.argtypes = [
    _types.handle,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
]


def cublasDtrmm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc):
    """Matrix-matrix product for real double precision triangular matrix"""
    assert _libcublas
    status = _libcublas.cublasDtrmm_v2(
        handle, _CUBLAS_SIDE_MODE[side], _CUBLAS_FILL_MODE[uplo], _CUBLAS_OP[trans], _CUBLAS_DIAG[diag], m, n, ctypes.byref(ctypes.c_double(alpha)), int(A), lda, int(B), ldb, int(C), ldc
    )
    cublasCheckStatus(status)

cublasDtrmm.__doc__ = _TRMM_doc.substitute(a_type="numpy.float64")

_libcublas.cublasCtrmm_v2.restype = int
_libcublas.cublasCtrmm_v2.argtypes = [
    _types.handle,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
]


def cublasCtrmm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc):
    """Matrix-matrix product for complex single precision triangular matrix"""
    assert _libcublas
    status = _libcublas.cublasCtrmm_v2(
        handle,
        _CUBLAS_SIDE_MODE[side],
        _CUBLAS_FILL_MODE[uplo],
        _CUBLAS_OP[trans],
        _CUBLAS_DIAG[diag],
        m,
        n,
        ctypes.byref(cuda.cuFloatComplex(alpha.real, alpha.imag)),
        int(A),
        lda,
        int(B),
        ldb,
        int(C),
        ldc
    )
    cublasCheckStatus(status)

cublasCtrmm.__doc__ = _TRMM_doc.substitute(a_type="numpy.complex64")

_libcublas.cublasZtrmm_v2.restype = int
_libcublas.cublasZtrmm_v2.argtypes = [
    _types.handle,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
]


def cublasZtrmm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc):
    """Matrix-matrix product for complex double precision triangular matrix"""
    assert _libcublas
    status = _libcublas.cublasZtrmm_v2(
        handle,
        _CUBLAS_SIDE_MODE[side],
        _CUBLAS_FILL_MODE[uplo],
        _CUBLAS_OP[trans],
        _CUBLAS_DIAG[diag],
        m,
        n,
        ctypes.byref(cuda.cuDoubleComplex(alpha.real, alpha.imag)),
        int(A),
        lda,
        int(B),
        ldb,
        int(C),
        ldc,
    )
    cublasCheckStatus(status)

cublasZtrmm.__doc__ = _TRMM_doc.substitute(a_type="numpy.complex128")

# STRSM, DTRSM, CTRSM, ZTRSM
_TRSM_doc = Template(
    """
    Solve a triangular system with multiple right-hand sides.

    Solves:
    `op(A)*X = alpha*B` if `side == 'L'`, or
    `X*op(A) = alpha*B` if `side == 'R'`.

    `A` is a triangular matrix. The solution overwrites `B`.

    Parameters
    ----------
    handle : int
        CUBLAS context.
    side : char
        Specifies whether the triangular matrix appears on the left
        or right side of the system:
        'L' or 'l': left side.
        'R' or 'r': right side.
    uplo : char
        Specifies whether `A` is upper or lower triangular:
        'U' or 'u': upper triangular.
        'L' or 'l': lower triangular.
    trans : char
        Specifies the form of `A`:
        'N' or 'n': no transpose.
        'T' or 't': transpose.
        'C' or 'c': conjugate transpose.
    diag : char
        Specifies whether `A` is unit triangular:
        'U' or 'u': unit triangular.
        'N' or 'n': non-unit triangular.
    m : int
        Number of rows of `B`.
    n : int
        Number of columns of `B`.
    alpha : ${a_type}
        Scalar multiplier for `B`.
    A : ctypes.c_void_p
        Pointer to triangular matrix `A`.
    lda : int
        Leading dimension of `A`.
    B : ctypes.c_void_p
        Pointer to matrix `B`. On exit, contains the solution `X`.
    ldb : int
        Leading dimension of `B`.

    References
    ----------
    `cublas<t>trsm <http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-trsm>`_
    """
)

_libcublas.cublasStrsm_v2.restype = int
_libcublas.cublasStrsm_v2.argtypes = [
    _types.handle,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
]


def cublasStrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb):
    """Solve a real single precision triangular system with multiple right-hand sides"""
    assert _libcublas
    status = _libcublas.cublasStrsm_v2(
        handle, _CUBLAS_SIDE_MODE[side], _CUBLAS_FILL_MODE[uplo], _CUBLAS_OP[trans], _CUBLAS_DIAG[diag], m, n, ctypes.byref(ctypes.c_float(alpha)), int(A), lda, int(B), ldb
    )
    cublasCheckStatus(status)

cublasStrsm.__doc__ = _TRSM_doc.substitute(a_type="numpy.float32")

_libcublas.cublasDtrsm_v2.restype = int
_libcublas.cublasDtrsm_v2.argtypes = [
    _types.handle,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
]


def cublasDtrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb):
    """Solve a real double precision triangular system with multiple right-hand sides"""
    assert _libcublas
    status = _libcublas.cublasDtrsm_v2(
        handle, _CUBLAS_SIDE_MODE[side], _CUBLAS_FILL_MODE[uplo], _CUBLAS_OP[trans], _CUBLAS_DIAG[diag], m, n, ctypes.byref(ctypes.c_double(alpha)), int(A), lda, int(B), ldb
    )
    cublasCheckStatus(status)

cublasDtrsm.__doc__ = _TRSM_doc.substitute(a_type="numpy.float64")

_libcublas.cublasCtrsm_v2.restype = int
_libcublas.cublasCtrsm_v2.argtypes = [
    _types.handle,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
]


def cublasCtrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb):
    """Solve a complex single precision triangular system with multiple right-hand sides"""
    assert _libcublas
    status = _libcublas.cublasCtrsm_v2(
        handle, _CUBLAS_SIDE_MODE[side], _CUBLAS_FILL_MODE[uplo], _CUBLAS_OP[trans], _CUBLAS_DIAG[diag], m, n, ctypes.byref(cuda.cuFloatComplex(alpha.real, alpha.imag)), int(A), lda, int(B), ldb
    )
    cublasCheckStatus(status)

cublasCtrsm.__doc__ = _TRSM_doc.substitute(a_type="numpy.complex64")

_libcublas.cublasZtrsm_v2.restype = int
_libcublas.cublasZtrsm_v2.argtypes = [
    _types.handle,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
]


def cublasZtrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb):
    """Solve complex double precision triangular system with multiple right-hand sides"""
    assert _libcublas
    status = _libcublas.cublasZtrsm_v2(
        handle, _CUBLAS_SIDE_MODE[side], _CUBLAS_FILL_MODE[uplo], _CUBLAS_OP[trans], _CUBLAS_DIAG[diag], m, n, ctypes.byref(cuda.cuDoubleComplex(alpha.real, alpha.imag)), int(A), lda, int(B), ldb
    )
    cublasCheckStatus(status)

cublasZtrsm.__doc__ = _TRSM_doc.substitute(a_type="numpy.complex128")

# CHEMM, ZHEMM
_HEMM_doc = Template(
    """
    Matrix-matrix multiplication with a Hermitian matrix.

    Computes:
    `C = alpha*A*B + beta*C` if `side == 'L'`, or
    `C = alpha*B*A + beta*C` if `side == 'R'`.

    `A` is a Hermitian matrix.

    Parameters
    ----------
    handle : int
        CUBLAS context.
    side : char
        Specifies whether the Hermitian matrix multiplies from the left
        or right:
        'L' or 'l': left side multiplication.
        'R' or 'r': right side multiplication.
    uplo : char
        Specifies whether the upper or lower triangular part of `A`
        is stored:
        'U' or 'u': upper triangular part is stored.
        'L' or 'l': lower triangular part is stored.
    m : int
        Number of rows of `C`.
    n : int
        Number of columns of `C`.
    alpha : ${a_type}
        Scalar multiplier for the matrix product.
    A : ctypes.c_void_p
        Pointer to Hermitian matrix `A`.
    lda : int
        Leading dimension of `A`.
    B : ctypes.c_void_p
        Pointer to matrix `B`.
    ldb : int
        Leading dimension of `B`.
    beta : ${a_type}
        Scalar multiplier for matrix `C`.
    C : ctypes.c_void_p
        Pointer to matrix `C`. The result is stored here.
    ldc : int
        Leading dimension of `C`.

    References
    ----------
    `cublas<t>hemm <http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-hemm>`_
    """
)

_libcublas.cublasChemm_v2.restype = int
_libcublas.cublasChemm_v2.argtypes = [
    _types.handle,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
]


def cublasChemm(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc):
    """Matrix-matrix product for single precision Hermitian matrix."""
    assert _libcublas
    status = _libcublas.cublasChemm_v2(
        handle,
        _CUBLAS_SIDE_MODE[side],
        _CUBLAS_FILL_MODE[uplo],
        m,
        n,
        ctypes.byref(cuda.cuFloatComplex(alpha.real, alpha.imag)),
        int(A),
        lda,
        int(B),
        ldb,
        ctypes.byref(cuda.cuFloatComplex(beta.real, beta.imag)),
        int(C),
        ldc,
    )
    cublasCheckStatus(status)

cublasChemm.__doc__ = _HEMM_doc.substitute(a_type="numpy.complex64")

_libcublas.cublasZhemm_v2.restype = int
_libcublas.cublasZhemm_v2.argtypes = [
    _types.handle,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
]


def cublasZhemm(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc):
    """Matrix-matrix product for double precision Hermitian matrix."""
    assert _libcublas
    status = _libcublas.cublasZhemm_v2(
        handle,
        _CUBLAS_SIDE_MODE[side],
        _CUBLAS_FILL_MODE[uplo],
        m,
        n,
        ctypes.byref(cuda.cuDoubleComplex(alpha.real, alpha.imag)),
        int(A),
        lda,
        int(B),
        ldb,
        ctypes.byref(cuda.cuDoubleComplex(beta.real, beta.imag)),
        int(C),
        ldc,
    )
    cublasCheckStatus(status)

cublasZhemm.__doc__ = _HEMM_doc.substitute(a_type="numpy.complex128")

# CHERK, ZHERK
_HERK_doc = Template(
    """
    Rank-k update of a Hermitian matrix.

    Updates the Hermitian matrix `C`:
    `C = alpha*op(A)*op(A)^H + beta*C`.

    Parameters
    ----------
    handle : int
        CUBLAS context.
    uplo : char
        Specifies whether the upper or lower triangular part of `C`
        is stored:
        'U' or 'u': upper triangular part is stored.
        'L' or 'l': lower triangular part is stored.
    trans : char
        Specifies the form of matrix `A`:
        'N' or 'n': no transpose.
        'T' or 't': transpose.
        'C' or 'c': conjugate transpose.
    n : int
        Number of rows and columns of `C`.
    k : int
        Inner dimension of the matrix product.
    alpha : ${scalar_type}
        Scalar multiplier for the rank-k product.
    A : ctypes.c_void_p
        Pointer to matrix `A`.
    lda : int
        Leading dimension of `A`.
    beta : ${scalar_type}
        Scalar multiplier for matrix `C`.
    C : ctypes.c_void_p
        Pointer to Hermitian matrix `C`. The result is stored here.
    ldc : int
        Leading dimension of `C`.

    References
    ----------
    `cublas<t>herk <http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-herk>`_
    """
)

_libcublas.cublasCherk_v2.restype = int
_libcublas.cublasCherk_v2.argtypes = [
    _types.handle,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
]


def cublasCherk(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc):
    """Rank-k operation on single precision Hermitian matrix."""
    assert _libcublas
    status = _libcublas.cublasCherk_v2(handle, _CUBLAS_FILL_MODE[uplo], _CUBLAS_OP[trans], n, k, ctypes.byref(ctypes.c_float(alpha)), int(A), lda, ctypes.byref(ctypes.c_float(beta)), int(C), ldc)
    cublasCheckStatus(status)

cublasCherk.__doc__ = _HERK_doc.substitute(scalar_type="numpy.float32")

_libcublas.cublasZherk_v2.restype = int
_libcublas.cublasZherk_v2.argtypes = [
    _types.handle,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
]


def cublasZherk(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc):
    """Rank-k operation on double precision Hermitian matrix."""
    assert _libcublas
    status = _libcublas.cublasZherk_v2(handle, _CUBLAS_FILL_MODE[uplo], _CUBLAS_OP[trans], n, k, ctypes.byref(ctypes.c_double(alpha)), int(A), lda, ctypes.byref(ctypes.c_double(beta)), int(C), ldc)
    cublasCheckStatus(status)

cublasZherk.__doc__ = _HERK_doc.substitute(scalar_type="numpy.float64")

# CHER2K, ZHER2K
_HER2K_doc = Template(
    """
    Rank-2k update of a Hermitian matrix.

    Updates the Hermitian matrix `C`:
    `C = alpha*op(A)*op(B)^H + conj(alpha)*op(B)*op(A)^H + beta*C`.

    Parameters
    ----------
    handle : int
        CUBLAS context.
    uplo : char
        Specifies whether the upper or lower triangular part of `C`
        is stored:
        'U' or 'u': upper triangular part is stored.
        'L' or 'l': lower triangular part is stored.
    trans : char
        Specifies the form of matrices `A` and `B`:
        'N' or 'n': no transpose.
        'T' or 't': transpose.
        'C' or 'c': conjugate transpose.
    n : int
        Number of rows and columns of `C`.
    k : int
        Inner dimension of the matrix products.
    alpha : ${alpha_type}
        Scalar multiplier for the rank-2k products.
    A : ctypes.c_void_p
        Pointer to matrix `A`.
    lda : int
        Leading dimension of `A`.
    B : ctypes.c_void_p
        Pointer to matrix `B`.
    ldb : int
        Leading dimension of `B`.
    beta : ${beta_type}
        Scalar multiplier for matrix `C`.
    C : ctypes.c_void_p
        Pointer to Hermitian matrix `C`. The result is stored here.
    ldc : int
        Leading dimension of `C`.

    References
    ----------
    `cublas<t>her2k <http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-her2k>`_
    """
)

_libcublas.cublasCher2k_v2.restype = int
_libcublas.cublasCher2k_v2.argtypes = [
    _types.handle,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_float,
    ctypes.c_void_p,
    ctypes.c_int,
]


def cublasCher2k(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
    """Rank-2k operation on single precision Hermitian matrix."""
    assert _libcublas
    status = _libcublas.cublasCher2k_v2(
        handle,
        _CUBLAS_FILL_MODE[uplo],
        _CUBLAS_OP[trans],
        n,
        k,
        ctypes.byref(cuda.cuFloatComplex(alpha.real, alpha.imag)),
        int(A),
        lda,
        int(B),
        ldb,
        ctypes.byref(cuda.cuFloatComplex(beta.real, beta.imag)),
        int(C),
        ldc,
    )
    cublasCheckStatus(status)

cublasCher2k.__doc__ = _HER2K_doc.substitute(
    alpha_type="numpy.complex64",
    beta_type="numpy.float32",
)

_libcublas.cublasZher2k_v2.restype = int
_libcublas.cublasZher2k_v2.argtypes = [
    _types.handle,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
]


def cublasZher2k(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
    """Rank-2k operation on double precision Hermitian matrix."""
    assert _libcublas
    status = _libcublas.cublasZher2k_v2(
        handle,
        _CUBLAS_FILL_MODE[uplo],
        _CUBLAS_OP[trans],
        n,
        k,
        ctypes.byref(cuda.cuDoubleComplex(alpha.real, alpha.imag)),
        int(A),
        lda,
        int(B),
        ldb,
        ctypes.byref(cuda.cuDoubleComplex(beta.real, beta.imag)),
        int(C),
        ldc,
    )
    cublasCheckStatus(status)

cublasZher2k.__doc__ = _HER2K_doc.substitute(
    alpha_type="numpy.complex128",
    beta_type="numpy.float64",
)

# BLAS-like extension routines


# SGEAM, DGEAM, CGEAM, ZGEAM
_GEAM_doc = Template(
    """
    Matrix-matrix addition/transposition (${precision} ${real}).

    Computes the sum of two ${precision} ${real} scaled and possibly (conjugate)
    transposed matrices.

    Parameters
    ----------
    handle : int
        CUBLAS context
    transa, transb : char
        't' if they are transposed, 'c' if they are conjugate transposed,
        'n' if otherwise.
    m : int
        Number of rows in `A` and `C`.
    n : int
        Number of columns in `B` and `C`.
    alpha : ${num_type}
        Constant by which to scale `A`.
    A : ctypes.c_void_p
        Pointer to first matrix operand (`A`).
    lda : int
        Leading dimension of `A`.
    beta : ${num_type}
        Constant by which to scale `B`.
    B : ctypes.c_void_p
        Pointer to second matrix operand (`B`).
    ldb : int
        Leading dimension of `B`.
    C : ctypes.c_void_p
        Pointer to result matrix (`C`).
    ldc : int
        Leading dimension of `C`.

    Examples
    --------
    >>> import pycuda.autoinit
    >>> import pycuda.gpuarray as gpuarray
    >>> import numpy as np
    >>> alpha = ${alpha_data}
    >>> beta = ${beta_data}
    >>> a = ${a_data_1}
    >>> b = ${b_data_1}
    >>> c = ${c_data_1}
    >>> a_gpu = gpuarray.to_gpu(a)
    >>> b_gpu = gpuarray.to_gpu(b)
    >>> c_gpu = gpuarray.empty(c.shape, c.dtype)
    >>> h = cublasCreate()
    >>> ${func}(h, 'n', 'n', c.shape[0], c.shape[1], alpha, a_gpu.gpudata, a.shape[0], beta, b_gpu.gpudata, b.shape[0], c_gpu.gpudata, c.shape[0])
    >>> np.allclose(c_gpu.get(), c)
    True
    >>> a = ${a_data_2}
    >>> b = ${b_data_2}
    >>> c = ${c_data_2}
    >>> a_gpu = gpuarray.to_gpu(a.T.copy())
    >>> b_gpu = gpuarray.to_gpu(b.T.copy())
    >>> c_gpu = gpuarray.empty(c.T.shape, c.dtype)
    >>> transa = 'c' if np.iscomplexobj(a) else 't'
    >>> ${func}(h, transa, 'n', c.shape[0], c.shape[1], alpha, a_gpu.gpudata, a.shape[0], beta, b_gpu.gpudata, b.shape[0], c_gpu.gpudata, c.shape[0])
    >>> np.allclose(c_gpu.get().T, c)
    True
    >>> cublasDestroy(h)

    References
    ----------
    `cublas<t>geam <http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-geam>`_
"""
)

if _cublas_version >= 5000:
    _libcublas.cublasSgeam.restype = int
    _libcublas.cublasSgeam.argtypes = [
        _types.handle,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
    ]


@_cublas_version_req(5.0)
def cublasSgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc):
    """Real matrix-matrix addition/transposition."""
    assert _libcublas
    status = _libcublas.cublasSgeam(
        handle, _CUBLAS_OP[transa], _CUBLAS_OP[transb], m, n, ctypes.byref(ctypes.c_float(alpha)), int(A), lda, ctypes.byref(ctypes.c_float(beta)), int(B), ldb, int(C), ldc
    )
    cublasCheckStatus(status)


cublasSgeam.__doc__ = _GEAM_doc.substitute(
    precision="single precision",
    real="real",
    num_type="numpy.float32",
    alpha_data="np.float32(np.random.rand())",
    beta_data="np.float32(np.random.rand())",
    a_data_1="np.random.rand(2, 3).astype(np.float32)",
    b_data_1="np.random.rand(2, 3).astype(np.float32)",
    a_data_2="np.random.rand(2, 3).astype(np.float32)",
    b_data_2="np.random.rand(3, 2).astype(np.float32)",
    c_data_1="alpha*a+beta*b",
    c_data_2="alpha*a.T+beta*b",
    func="cublasSgeam",
)

if _cublas_version >= 5000:
    _libcublas.cublasDgeam.restype = int
    _libcublas.cublasDgeam.argtypes = [
        _types.handle,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
    ]


@_cublas_version_req(5.0)
def cublasDgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc):
    """Real matrix-matrix addition/transposition."""
    assert _libcublas
    status = _libcublas.cublasDgeam(
        handle, _CUBLAS_OP[transa], _CUBLAS_OP[transb], m, n, ctypes.byref(ctypes.c_double(alpha)), int(A), lda, ctypes.byref(ctypes.c_double(beta)), int(B), ldb, int(C), ldc
    )
    cublasCheckStatus(status)


cublasDgeam.__doc__ = _GEAM_doc.substitute(
    precision="double precision",
    real="real",
    num_type="numpy.float64",
    alpha_data="np.float64(np.random.rand())",
    beta_data="np.float64(np.random.rand())",
    a_data_1="np.random.rand(2, 3).astype(np.float64)",
    b_data_1="np.random.rand(2, 3).astype(np.float64)",
    a_data_2="np.random.rand(2, 3).astype(np.float64)",
    b_data_2="np.random.rand(3, 2).astype(np.float64)",
    c_data_1="alpha*a+beta*b",
    c_data_2="alpha*a.T+beta*b",
    func="cublasDgeam",
)

if _cublas_version >= 5000:
    _libcublas.cublasCgeam.restype = int
    _libcublas.cublasCgeam.argtypes = [
        _types.handle,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
    ]


@_cublas_version_req(5.0)
def cublasCgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc):
    """Complex matrix-matrix addition/transposition."""
    assert _libcublas
    status = _libcublas.cublasCgeam(
        handle,
        _CUBLAS_OP[transa],
        _CUBLAS_OP[transb],
        m,
        n,
        ctypes.byref(cuda.cuFloatComplex(alpha.real, alpha.imag)),
        int(A),
        lda,
        ctypes.byref(cuda.cuFloatComplex(beta.real, beta.imag)),
        int(B),
        ldb,
        int(C),
        ldc,
    )
    cublasCheckStatus(status)


cublasCgeam.__doc__ = _GEAM_doc.substitute(
    precision="single precision",
    real="complex",
    num_type="numpy.complex64",
    alpha_data="np.complex64(np.random.rand()+1j*np.random.rand())",
    beta_data="np.complex64(np.random.rand()+1j*np.random.rand())",
    a_data_1="(np.random.rand(2, 3)+1j*np.random.rand(2, 3)).astype(np.complex64)",
    a_data_2="(np.random.rand(2, 3)+1j*np.random.rand(2, 3)).astype(np.complex64)",
    b_data_1="(np.random.rand(2, 3)+1j*np.random.rand(2, 3)).astype(np.complex64)",
    b_data_2="(np.random.rand(3, 2)+1j*np.random.rand(3, 2)).astype(np.complex64)",
    c_data_1="alpha*a+beta*b",
    c_data_2="alpha*np.conj(a).T+beta*b",
    func="cublasCgeam",
)

if _cublas_version >= 5000:
    _libcublas.cublasZgeam.restype = int
    _libcublas.cublasZgeam.argtypes = [
        _types.handle,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
    ]


@_cublas_version_req(5.0)
def cublasZgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc):
    """Complex matrix-matrix addition/transposition."""
    assert _libcublas
    status = _libcublas.cublasZgeam(
        handle,
        _CUBLAS_OP[transa],
        _CUBLAS_OP[transb],
        m,
        n,
        ctypes.byref(cuda.cuDoubleComplex(alpha.real, alpha.imag)),
        int(A),
        lda,
        ctypes.byref(cuda.cuDoubleComplex(beta.real, beta.imag)),
        int(B),
        ldb,
        int(C),
        ldc,
    )
    cublasCheckStatus(status)


cublasZgeam.__doc__ = _GEAM_doc.substitute(
    precision="double precision",
    real="complex",
    num_type="numpy.complex128",
    alpha_data="np.complex128(np.random.rand()+1j*np.random.rand())",
    beta_data="np.complex128(np.random.rand()+1j*np.random.rand())",
    a_data_1="(np.random.rand(2, 3)+1j*np.random.rand(2, 3)).astype(np.complex128)",
    a_data_2="(np.random.rand(2, 3)+1j*np.random.rand(2, 3)).astype(np.complex128)",
    b_data_1="(np.random.rand(2, 3)+1j*np.random.rand(2, 3)).astype(np.complex128)",
    b_data_2="(np.random.rand(3, 2)+1j*np.random.rand(3, 2)).astype(np.complex128)",
    c_data_1="alpha*a+beta*b",
    c_data_2="alpha*np.conj(a).T+beta*b",
    func="cublasZgeam",
)

# Batched routines

# SgemmBatched, DgemmBatched, CgemmBatched, ZgemmBatched
_GEMM_BATCHED_doc = Template(
    """
    Batched matrix-matrix multiplication (${precision} ${real}).

    Computes a batch of matrix products:

    `C_i = alpha * op(A_i) * op(B_i) + beta * C_i`

    where `op(X)` is `X`, `X.T`, or `X.H` depending on the transpose mode.

    Parameters
    ----------
    handle : int
        CUBLAS context.
    transa, transb : char
        't' if transposed, 'c' if conjugate transposed, 'n' otherwise.
    m : int
        Number of rows of `op(A)` and `C`.
    n : int
        Number of columns of `op(B)` and `C`.
    k : int
        Number of columns of `op(A)` and rows of `op(B)`.
    alpha : ${num_type}
        Scalar multiplier for `A @ B`.
    A : ctypes.c_void_p
        Device pointer to array of pointers to matrices `A`.
    lda : int
        Leading dimension of `A`.
    B : ctypes.c_void_p
        Device pointer to array of pointers to matrices `B`.
    ldb : int
        Leading dimension of `B`.
    beta : ${num_type}
        Scalar multiplier for `C`.
    C : ctypes.c_void_p
        Device pointer to array of pointers to matrices `C`.
    ldc : int
        Leading dimension of `C`.
    batchCount : int
        Number of matrices in the batch.

    References
    ----------
    `cublas<t>gemmBatched <http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemmbatched>`_
"""
)

if _cublas_version >= 5000:
    _libcublas.cublasSgemmBatched.restype = int
    _libcublas.cublasSgemmBatched.argtypes = [
        _types.handle,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
    ]


@_cublas_version_req(5.0)
def cublasSgemmBatched(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batchCount):
    """Matrix-matrix product for arrays of real single precision general matrices."""

    assert _libcublas
    status = _libcublas.cublasSgemmBatched(
        handle, _CUBLAS_OP[transa], _CUBLAS_OP[transb], m, n, k, ctypes.byref(ctypes.c_float(alpha)), int(A), lda, int(B), ldb, ctypes.byref(ctypes.c_float(beta)), int(C), ldc, batchCount
    )
    cublasCheckStatus(status)

cublasSgemmBatched.__doc__ = _GEMM_BATCHED_doc.substitute(
    precision="single precision",
    real="real",
    num_type="numpy.float32",
)

if _cublas_version >= 5000:
    _libcublas.cublasDgemmBatched.restype = int
    _libcublas.cublasDgemmBatched.argtypes = [
        _types.handle,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
    ]


@_cublas_version_req(5.0)
def cublasDgemmBatched(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batchCount):
    """Matrix-matrix product for arrays of real double precision general matrices."""

    assert _libcublas
    status = _libcublas.cublasDgemmBatched(
        handle, _CUBLAS_OP[transa], _CUBLAS_OP[transb], m, n, k, ctypes.byref(ctypes.c_double(alpha)), int(A), lda, int(B), ldb, ctypes.byref(ctypes.c_double(beta)), int(C), ldc, batchCount
    )
    cublasCheckStatus(status)

cublasDgemmBatched.__doc__ = _GEMM_BATCHED_doc.substitute(
    precision="double precision",
    real="real",
    num_type="numpy.float64",
)

if _cublas_version >= 5000:
    _libcublas.cublasCgemmBatched.restype = int
    _libcublas.cublasCgemmBatched.argtypes = [
        _types.handle,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
    ]


@_cublas_version_req(5.0)
def cublasCgemmBatched(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batchCount):
    """Matrix-matrix product for arrays of complex single precision general matrices."""

    assert _libcublas
    status = _libcublas.cublasCgemmBatched(
        handle,
        _CUBLAS_OP[transa],
        _CUBLAS_OP[transb],
        m,
        n,
        k,
        ctypes.byref(cuda.cuFloatComplex(alpha.real, alpha.imag)),
        int(A),
        lda,
        int(B),
        ldb,
        ctypes.byref(cuda.cuFloatComplex(beta.real, beta.imag)),
        int(C),
        ldc,
        batchCount,
    )
    cublasCheckStatus(status)

cublasCgemmBatched.__doc__ = _GEMM_BATCHED_doc.substitute(
    precision="single precision",
    real="complex",
    num_type="numpy.complex64",
)

if _cublas_version >= 5000:
    _libcublas.cublasZgemmBatched.restype = int
    _libcublas.cublasZgemmBatched.argtypes = [
        _types.handle,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
    ]


@_cublas_version_req(5.0)
def cublasZgemmBatched(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batchCount):
    """Matrix-matrix product for arrays of complex double precision general matrices."""

    assert _libcublas
    status = _libcublas.cublasZgemmBatched(
        handle,
        _CUBLAS_OP[transa],
        _CUBLAS_OP[transb],
        m,
        n,
        k,
        ctypes.byref(cuda.cuDoubleComplex(alpha.real, alpha.imag)),
        int(A),
        lda,
        int(B),
        ldb,
        ctypes.byref(cuda.cuDoubleComplex(beta.real, beta.imag)),
        int(C),
        ldc,
        batchCount,
    )
    cublasCheckStatus(status)

cublasZgemmBatched.__doc__ = _GEMM_BATCHED_doc.substitute(
    precision="double precision",
    real="complex",
    num_type="numpy.complex128",
)

# StrsmBatched, DtrsmBatched
_TRSM_BATCHED_doc = Template(
    """
    Batched triangular solve (${precision} ${real}).

    Solves a batch of triangular linear systems with multiple right-hand sides.

    Depending on `side`, solves one of:

    `op(A_i) * X_i = alpha * B_i`

    or

    `X_i * op(A_i) = alpha * B_i`

    Parameters
    ----------
    handle : int
        CUBLAS context.
    side : char
        'l' if `A` multiplies from the left, 'r' if from the right.
    uplo : char
        'u' if `A` is upper triangular, 'l' if lower triangular.
    trans : char
        't' if transposed, 'c' if conjugate transposed, 'n' otherwise.
    diag : char
        'u' if `A` is unit triangular, 'n' otherwise.
    m : int
        Number of rows of `B`.
    n : int
        Number of columns of `B`.
    alpha : ${num_type}
        Scalar multiplier for `B`.
    A : ctypes.c_void_p
        Device pointer to array of pointers to triangular matrices `A`.
    lda : int
        Leading dimension of `A`.
    B : ctypes.c_void_p
        Device pointer to array of pointers to right-hand side matrices `B`.
    ldb : int
        Leading dimension of `B`.
    batchCount : int
        Number of systems in the batch.

    References
    ----------
    `cublas<t>trsmBatched <http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-trsmbatched>`_
"""
)

if _cublas_version >= 5000:
    _libcublas.cublasStrsmBatched.restype = int
    _libcublas.cublasStrsmBatched.argtypes = [
        _types.handle,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
    ]


@_cublas_version_req(5.0)
def cublasStrsmBatched(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount):
    """This function solves an array of triangular linear systems with multiple right-hand-sides."""

    assert _libcublas
    status = _libcublas.cublasStrsmBatched(
        handle, _CUBLAS_SIDE_MODE[side], _CUBLAS_FILL_MODE[uplo], _CUBLAS_OP[trans], _CUBLAS_DIAG[diag], m, n, ctypes.byref(ctypes.c_float(alpha)), int(A), lda, int(B), ldb, batchCount
    )
    cublasCheckStatus(status)

cublasStrsmBatched.__doc__ = _TRSM_BATCHED_doc.substitute(
    precision="single precision",
    real="real",
    num_type="numpy.float32",
)

if _cublas_version >= 5000:
    _libcublas.cublasDtrsmBatched.restype = int
    _libcublas.cublasDtrsmBatched.argtypes = [
        _types.handle,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
    ]


@_cublas_version_req(5.0)
def cublasDtrsmBatched(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount):
    """This function solves an array of triangular linear systems with multiple right-hand-sides."""

    assert _libcublas
    status = _libcublas.cublasDtrsmBatched(
        handle, _CUBLAS_SIDE_MODE[side], _CUBLAS_FILL_MODE[uplo], _CUBLAS_OP[trans], _CUBLAS_DIAG[diag], m, n, ctypes.byref(ctypes.c_double(alpha)), int(A), lda, int(B), ldb, batchCount
    )
    cublasCheckStatus(status)

cublasDtrsmBatched.__doc__ = _TRSM_BATCHED_doc.substitute(
    precision="double precision",
    real="real",
    num_type="numpy.float64",
)

# SgetrfBatched, DgetrfBatched,CgetrfBatched, ZgetrfBatched
_GETRF_BATCHED_doc = Template(
    """
    Batched LU factorization (${precision} ${real}).

    Computes the LU factorization of a batch of square matrices using
    partial pivoting:

    `P_i * A_i = L_i * U_i`

    Parameters
    ----------
    handle : int
        CUBLAS context.
    n : int
        Order of the matrices.
    A : ctypes.c_void_p
        Device pointer to array of pointers to matrices to be factorized.
        On output, contains the combined `L` and `U` factors.
    lda : int
        Leading dimension of `A`.
    P : ctypes.c_void_p
        Device pointer to pivot arrays.
    info : ctypes.c_void_p
        Device pointer to info array containing factorization status.
    batchSize : int
        Number of matrices in the batch.

    References
    ----------
    `cublas<t>getrfBatched <http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-getrfbatched>`_
"""
)

if _cublas_version >= 5000:
    _libcublas.cublasSgetrfBatched.restype = int
    _libcublas.cublasSgetrfBatched.argtypes = [_types.handle, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]


@_cublas_version_req(5.0)
def cublasSgetrfBatched(handle, n, A, lda, P, info, batchSize):
    """This function performs the LU factorization of an array of n x n matrices."""

    assert _libcublas
    status = _libcublas.cublasSgetrfBatched(handle, n, int(A), lda, int(P), int(info), batchSize)
    cublasCheckStatus(status)

cublasSgetrfBatched.__doc__ = _GETRF_BATCHED_doc.substitute(
    precision="single precision",
    real="real",
)

if _cublas_version >= 5000:
    _libcublas.cublasDgetrfBatched.restype = int
    _libcublas.cublasDgetrfBatched.argtypes = [_types.handle, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]


@_cublas_version_req(5.0)
def cublasDgetrfBatched(handle, n, A, lda, P, info, batchSize):
    """This function performs the LU factorization of an array of n x n matrices."""

    assert _libcublas
    status = _libcublas.cublasDgetrfBatched(handle, n, int(A), lda, int(P), int(info), batchSize)
    cublasCheckStatus(status)

cublasDgetrfBatched.__doc__ = _GETRF_BATCHED_doc.substitute(
    precision="double precision",
    real="real",
)

if _cublas_version >= 5000:
    _libcublas.cublasCgetrfBatched.restype = int
    _libcublas.cublasCgetrfBatched.argtypes = [_types.handle, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]


@_cublas_version_req(5.0)
def cublasCgetrfBatched(handle, n, A, lda, P, info, batchSize):
    """This function performs the LU factorization of an array of n x n matrices."""

    assert _libcublas
    status = _libcublas.cublasCgetrfBatched(handle, n, int(A), lda, int(P), int(info), batchSize)
    cublasCheckStatus(status)

cublasCgetrfBatched.__doc__ = _GETRF_BATCHED_doc.substitute(
    precision="single precision",
    real="complex",
)

if _cublas_version >= 5000:
    _libcublas.cublasZgetrfBatched.restype = int
    _libcublas.cublasZgetrfBatched.argtypes = [_types.handle, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]


@_cublas_version_req(5.0)
def cublasZgetrfBatched(handle, n, A, lda, P, info, batchSize):
    """This function performs the LU factorization of an array of n x n matrices."""

    assert _libcublas
    status = _libcublas.cublasZgetrfBatched(handle, n, int(A), lda, int(P), int(info), batchSize)
    cublasCheckStatus(status)

cublasZgetrfBatched.__doc__ = _GETRF_BATCHED_doc.substitute(
    precision="double precision",
    real="complex",
)

# SgetrsBatched, DgetrsBatched, CgetrsBatched, ZgetrsBatched
_GETRS_BATCHED_doc = Template(
    """
    Batched LU-based linear solve (${precision} ${real}).

    Solves a batch of linear systems using matrices previously factorized
    with `getrfBatched`.

    Parameters
    ----------
    handle : int
        CUBLAS context.
    trans : char
        't' if transposed, 'c' if conjugate transposed, 'n' otherwise.
    n : int
        Order of the matrices.
    nrhs : int
        Number of right-hand sides.
    Aarray : ctypes.c_void_p
        Device pointer to array of pointers to LU-factorized matrices.
    lda : int
        Leading dimension of `Aarray`.
    devIpiv : ctypes.c_void_p
        Device pointer to pivot arrays.
    Barray : ctypes.c_void_p
        Device pointer to array of pointers to right-hand side matrices.
        Overwritten with the solution matrices.
    ldb : int
        Leading dimension of `Barray`.
    info : ctypes.c_void_p
        Device pointer to info array containing solve status.
    batchSize : int
        Number of systems in the batch.

    References
    ----------
    `cublas<t>getrsBatched <https://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-getrsbatched>`_
"""
)

if _cublas_version >= 5000:
    _libcublas.cublasSgetrsBatched.restype = int
    _libcublas.cublasSgetrsBatched.argtypes = [
        _types.handle,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
    ]


@_cublas_version_req(5.0)
def cublasSgetrsBatched(handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize):
    """This function solves an array of LU factored linear systems."""

    assert _libcublas
    status = _libcublas.cublasSgetrsBatched(handle, _CUBLAS_OP[trans], n, nrhs, int(Aarray), lda, int(devIpiv), int(Barray), ldb, info, batchSize)
    cublasCheckStatus(status)

cublasSgetrsBatched.__doc__ = _GETRS_BATCHED_doc.substitute(
    precision="single precision",
    real="real",
)

if _cublas_version >= 5000:
    _libcublas.cublasDgetrsBatched.restype = int
    _libcublas.cublasDgetrsBatched.argtypes = [
        _types.handle,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
    ]


@_cublas_version_req(5.0)
def cublasDgetrsBatched(handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize):
    """This function solves an array of LU factored linear systems."""

    assert _libcublas
    status = _libcublas.cublasDgetrsBatched(handle, _CUBLAS_OP[trans], n, nrhs, int(Aarray), lda, int(devIpiv), int(Barray), ldb, info, batchSize)
    cublasCheckStatus(status)

cublasDgetrsBatched.__doc__ = _GETRS_BATCHED_doc.substitute(
    precision="double precision",
    real="real",
)

if _cublas_version >= 5000:
    _libcublas.cublasCgetrsBatched.restype = int
    _libcublas.cublasCgetrsBatched.argtypes = [
        _types.handle,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
    ]


@_cublas_version_req(5.0)
def cublasCgetrsBatched(handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize):
    """This function solves an array of LU factored linear systems."""

    assert _libcublas
    status = _libcublas.cublasCgetrsBatched(handle, _CUBLAS_OP[trans], n, nrhs, int(Aarray), lda, int(devIpiv), int(Barray), ldb, info, batchSize)
    cublasCheckStatus(status)

cublasCgetrsBatched.__doc__ = _GETRS_BATCHED_doc.substitute(
    precision="single precision",
    real="complex",
)

if _cublas_version >= 5000:
    _libcublas.cublasZgetrsBatched.restype = int
    _libcublas.cublasZgetrsBatched.argtypes = [
        _types.handle,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
    ]


@_cublas_version_req(5.0)
def cublasZgetrsBatched(handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize):
    """This function solves an array of LU factored linear systems."""

    assert _libcublas
    status = _libcublas.cublasZgetrsBatched(handle, _CUBLAS_OP[trans], n, nrhs, int(Aarray), lda, int(devIpiv), int(Barray), ldb, info, batchSize)
    cublasCheckStatus(status)

cublasZgetrsBatched.__doc__ = _GETRS_BATCHED_doc.substitute(
    precision="double precision",
    real="complex",
)

# SgetriBatched, DgetriBatched, CgetriBatched, ZgetriBatched
_GETRI_BATCHED_doc = Template(
    """
    Batched matrix inversion (${precision} ${real}).

    Computes the inverse of a batch of matrices previously factorized
    with `getrfBatched`.

    Parameters
    ----------
    handle : int
        CUBLAS context.
    n : int
        Order of the matrices.
    A : ctypes.c_void_p
        Device pointer to array of pointers to LU-factorized matrices.
    lda : int
        Leading dimension of `A`.
    P : ctypes.c_void_p
        Device pointer to pivot arrays.
    C : ctypes.c_void_p
        Device pointer to array of pointers to output inverse matrices.
    ldc : int
        Leading dimension of `C`.
    info : ctypes.c_void_p
        Device pointer to info array containing inversion status.
    batchSize : int
        Number of matrices in the batch.

    Notes
    -----
    The matrices must first be factorized using `${getrf_func}`.

    References
    ----------
    `cublas<t>getriBatched <http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-getribatched>`_
"""
)

if _cublas_version >= 5050:
    _libcublas.cublasSgetriBatched.restype = int
    _libcublas.cublasSgetriBatched.argtypes = [_types.handle, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]


@_cublas_version_req(5.5)
def cublasSgetriBatched(handle, n, A, lda, P, C, ldc, info, batchSize):
    """This function performs the inversion of an array of n x n matrices."""

    assert _libcublas
    status = _libcublas.cublasSgetriBatched(handle, n, int(A), lda, int(P), int(C), ldc, int(info), batchSize)
    cublasCheckStatus(status)

cublasSgetriBatched.__doc__ = _GETRI_BATCHED_doc.substitute(
    precision="single precision",
    real="real",
    getrf_func="cublasSgetrfBatched",
)

if _cublas_version >= 5050:
    _libcublas.cublasDgetriBatched.restype = int
    _libcublas.cublasDgetriBatched.argtypes = [_types.handle, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]


@_cublas_version_req(5.5)
def cublasDgetriBatched(handle, n, A, lda, P, C, ldc, info, batchSize):
    """This function performs the inversion of an array of n x n matrices."""

    assert _libcublas
    status = _libcublas.cublasDgetriBatched(handle, n, int(A), lda, int(P), int(C), ldc, int(info), batchSize)
    cublasCheckStatus(status)

cublasDgetriBatched.__doc__ = _GETRI_BATCHED_doc.substitute(
    precision="double precision",
    real="real",
    getrf_func="cublasDgetrfBatched",
)

if _cublas_version >= 5050:
    _libcublas.cublasCgetriBatched.restype = int
    _libcublas.cublasCgetriBatched.argtypes = [_types.handle, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]


@_cublas_version_req(5.5)
def cublasCgetriBatched(handle, n, A, lda, P, C, ldc, info, batchSize):
    """This function performs the inversion of an array of n x n matrices."""

    assert _libcublas
    status = _libcublas.cublasCgetriBatched(handle, n, int(A), lda, int(P), int(C), ldc, int(info), batchSize)
    cublasCheckStatus(status)

cublasCgetriBatched.__doc__ = _GETRI_BATCHED_doc.substitute(
    precision="single precision",
    real="complex",
    getrf_func="cublasCgetrfBatched",
)

if _cublas_version >= 5050:
    _libcublas.cublasZgetriBatched.restype = int
    _libcublas.cublasZgetriBatched.argtypes = [_types.handle, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]


@_cublas_version_req(5.5)
def cublasZgetriBatched(handle, n, A, lda, P, C, ldc, info, batchSize):
    """This function performs the inversion of an array of n x n matrices."""

    assert _libcublas
    status = _libcublas.cublasZgetriBatched(handle, n, int(A), lda, int(P), int(C), ldc, int(info), batchSize)
    cublasCheckStatus(status)

cublasZgetriBatched.__doc__ = _GETRI_BATCHED_doc.substitute(
    precision="double precision",
    real="complex",
    getrf_func="cublasZgetrfBatched",
)

# SgelsBatched, DgelsBatched, CgelsBatched, ZgelsBatched
_GELS_BATCHED_doc = Template(
    """
    Batched least-squares solver (${precision} ${real}).

    Computes the least-squares solution of a batch of overdetermined
    linear systems.

    Parameters
    ----------
    handle : int
        CUBLAS context.
    trans : char
        't' if transposed, 'c' if conjugate transposed, 'n' otherwise.
    m : int
        Number of rows of `A`.
    n : int
        Number of columns of `A`.
    nrhs : int
        Number of right-hand sides.
    Aarray : ctypes.c_void_p
        Device pointer to array of pointers to coefficient matrices.
    lda : int
        Leading dimension of `Aarray`.
    Carray : ctypes.c_void_p
        Device pointer to array of pointers to right-hand side matrices.
        Overwritten with the solution vectors.
    ldc : int
        Leading dimension of `Carray`.
    info : ctypes.c_void_p
        Host pointer returning execution status.
    devInfoArray : ctypes.c_void_p
        Device pointer to per-system status information.
    batchSize : int
        Number of systems in the batch.

    References
    ----------
    `cublas<t>gelsBatched <http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gelsbatched>`_
"""
)

if _cublas_version >= 5000:
    _libcublas.cublasSgelsBatched.restype = _libcublas.cublasDgelsBatched.restype = _libcublas.cublasCgelsBatched.restype = _libcublas.cublasZgelsBatched.restype = int
    _libcublas.cublasSgelsBatched.argtypes = _libcublas.cublasDgelsBatched.argtypes = _libcublas.cublasCgelsBatched.argtypes = _libcublas.cublasZgelsBatched.argtypes = [
        _types.handle,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
    ]


@_cublas_version_req(5.0)
def cublasSgelsBatched(handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize):
    """This function finds the least squares solution of a batch of overdetermined systems."""

    assert _libcublas
    status = _libcublas.cublasSgelsBatched(handle, _CUBLAS_OP[trans], m, n, nrhs, int(Aarray), lda, int(Carray), ldc, info, int(devInfoArray), batchSize)
    cublasCheckStatus(status)

cublasSgelsBatched.__doc__ = _GELS_BATCHED_doc.substitute(
    precision="single precision",
    real="real",
)

@_cublas_version_req(5.0)
def cublasDgelsBatched(handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize):
    """This function finds the least squares solution of a batch of overdetermined systems."""

    assert _libcublas
    status = _libcublas.cublasDgelsBatched(handle, _CUBLAS_OP[trans], m, n, nrhs, int(Aarray), lda, int(Carray), ldc, info, int(devInfoArray), batchSize)
    cublasCheckStatus(status)

cublasDgelsBatched.__doc__ = _GELS_BATCHED_doc.substitute(
    precision="double precision",
    real="real",
)

@_cublas_version_req(5.0)
def cublasCgelsBatched(handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize):
    """This function finds the least squares solution of a batch of overdetermined systems."""

    assert _libcublas
    status = _libcublas.cublasCgelsBatched(handle, _CUBLAS_OP[trans], m, n, nrhs, int(Aarray), lda, int(Carray), ldc, info, int(devInfoArray), batchSize)
    cublasCheckStatus(status)

cublasCgelsBatched.__doc__ = _GELS_BATCHED_doc.substitute(
    precision="single precision",
    real="complex",
)

@_cublas_version_req(5.0)
def cublasZgelsBatched(handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize):
    """This function finds the least squares solution of a batch of overdetermined systems."""

    assert _libcublas
    status = _libcublas.cublasZgelsBatched(handle, _CUBLAS_OP[trans], m, n, nrhs, int(Aarray), lda, int(Carray), ldc, info, int(devInfoArray), batchSize)
    cublasCheckStatus(status)

cublasZgelsBatched.__doc__ = _GELS_BATCHED_doc.substitute(
    precision="double precision",
    real="complex",
)

if _cublas_version >= 5000:
    _libcublas.cublasSdgmm.restype = _libcublas.cublasDdgmm.restype = _libcublas.cublasCdgmm.restype = _libcublas.cublasZdgmm.restype = int

    _libcublas.cublasSdgmm.argtypes = _libcublas.cublasDdgmm.argtypes = _libcublas.cublasCdgmm.argtypes = _libcublas.cublasZdgmm.argtypes = [
        _types.handle,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
    ]

# SDGMM, DDGMM, CDGMM, ZDGMM
_DGMM_doc = Template(
    """
    Matrix-diagonal matrix multiplication (${precision} ${real}).

    Multiplies a matrix by a diagonal matrix formed from a vector.

    Depending on `side`, computes one of:

    `C = diag(x) * A`

    or

    `C = A * diag(x)`

    Parameters
    ----------
    handle : int
        CUBLAS context.
    side : char
        'l' for left multiplication, 'r' for right multiplication.
    m : int
        Number of rows of `A` and `C`.
    n : int
        Number of columns of `A` and `C`.
    A : ctypes.c_void_p
        Pointer to input matrix `A`.
    lda : int
        Leading dimension of `A`.
    x : ctypes.c_void_p
        Pointer to vector defining the diagonal matrix.
    incx : int
        Increment for elements of `x`.
    C : ctypes.c_void_p
        Pointer to output matrix `C`.
    ldc : int
        Leading dimension of `C`.

    References
    ----------
    `cublas<t>dgmm <http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-dgmm>`_
"""
)

@_cublas_version_req(5.0)
def cublasSdgmm(handle, side, m, n, A, lda, x, incx, C, ldc):
    """Multiplies a matrix with a diagonal matrix."""

    assert _libcublas
    status = _libcublas.cublasSdgmm(handle, _CUBLAS_SIDE_MODE[side], m, n, int(A), lda, int(x), incx, int(C), ldc)
    cublasCheckStatus(status)

cublasSdgmm.__doc__ = _DGMM_doc.substitute(
    precision="single precision",
    real="real",
)

@_cublas_version_req(5.0)
def cublasDdgmm(handle, side, m, n, A, lda, x, incx, C, ldc):
    """Multiplies a matrix with a diagonal matrix."""

    assert _libcublas
    status = _libcublas.cublasDdgmm(handle, _CUBLAS_SIDE_MODE[side], m, n, int(A), lda, int(x), incx, int(C), ldc)
    cublasCheckStatus(status)

cublasDdgmm.__doc__ = _DGMM_doc.substitute(
    precision="double precision",
    real="real",
)

@_cublas_version_req(5.0)
def cublasCdgmm(handle, side, m, n, A, lda, x, incx, C, ldc):
    """Multiplies a matrix with a diagonal matrix."""

    assert _libcublas
    status = _libcublas.cublasCdgmm(handle, _CUBLAS_SIDE_MODE[side], m, n, int(A), lda, int(x), incx, int(C), ldc)
    cublasCheckStatus(status)

cublasCdgmm.__doc__ = _DGMM_doc.substitute(
    precision="single precision",
    real="complex",
)

@_cublas_version_req(5.0)
def cublasZdgmm(handle, side, m, n, A, lda, x, incx, C, ldc):
    """Multiplies a matrix with a diagonal matrix."""

    assert _libcublas
    status = _libcublas.cublasZdgmm(handle, _CUBLAS_SIDE_MODE[side], m, n, int(A), lda, int(x), incx, int(C), ldc)
    cublasCheckStatus(status)

cublasZdgmm.__doc__ = _DGMM_doc.substitute(
    precision="double precision",
    real="complex",
)

# SGEMMSTRIDEDBATCHED, DGEMMSTRIDEDBATCHED,
# CGEMMSTRIDEDBATCHED, ZGEMMSTRIDEDBATCHED
_GEMM_STRIDED_BATCHED_doc = Template(
    """
    Strided batched matrix-matrix multiplication (${precision} ${real}).

    Computes a batch of matrix products using regularly strided memory:

    `C_i = alpha * op(A_i) * op(B_i) + beta * C_i`

    Parameters
    ----------
    handle : int
        CUBLAS context.
    transa, transb : char
        't' if transposed, 'c' if conjugate transposed, 'n' otherwise.
    m : int
        Number of rows of `op(A)` and `C`.
    n : int
        Number of columns of `op(B)` and `C`.
    k : int
        Number of columns of `op(A)` and rows of `op(B)`.
    alpha : ${num_type}
        Scalar multiplier for `A @ B`.
    A : ctypes.c_void_p
        Pointer to first matrix batch.
    lda : int
        Leading dimension of `A`.
    strideA : int
        Stride between consecutive matrices in `A`.
    B : ctypes.c_void_p
        Pointer to second matrix batch.
    ldb : int
        Leading dimension of `B`.
    strideB : int
        Stride between consecutive matrices in `B`.
    beta : ${num_type}
        Scalar multiplier for `C`.
    C : ctypes.c_void_p
        Pointer to output matrix batch.
    ldc : int
        Leading dimension of `C`.
    strideC : int
        Stride between consecutive matrices in `C`.
    batchCount : int
        Number of matrices in the batch.

    References
    ----------
    `cublas<t>gemmStridedBatched <https://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemmstridedbatched>`_
"""
)

if _cublas_version >= 8000:
    _libcublas.cublasSgemmStridedBatched.restype = _libcublas.cublasDgemmStridedBatched.restype = _libcublas.cublasCgemmStridedBatched.restype = _libcublas.cublasZgemmStridedBatched.restype = int

    _libcublas.cublasSgemmStridedBatched.argtypes = _libcublas.cublasDgemmStridedBatched.argtypes = _libcublas.cublasCgemmStridedBatched.argtypes = _libcublas.cublasZgemmStridedBatched.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_longlong,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_longlong,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_longlong,
        ctypes.c_int,
    ]


@_cublas_version_req(8.0)
def cublasSgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount):
    """Matrix-matrix multiplication of a batch of matrices."""

    assert _libcublas
    status = _libcublas.cublasSgemmStridedBatched(
        handle,
        _CUBLAS_OP[transa],
        _CUBLAS_OP[transb],
        m,
        n,
        k,
        ctypes.byref(ctypes.c_float(alpha)),
        int(A),
        lda,
        strideA,
        int(B),
        ldb,
        strideB,
        ctypes.byref(ctypes.c_float(beta)),
        int(C),
        ldc,
        strideC,
        batchCount,
    )
    cublasCheckStatus(status)

cublasSgemmStridedBatched.__doc__ = _GEMM_STRIDED_BATCHED_doc.substitute(
    precision="single precision",
    real="real",
    num_type="numpy.float32",
)

@_cublas_version_req(8.0)
def cublasDgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount):
    """Matrix-matrix multiplication of a batch of matrices."""

    assert _libcublas
    status = _libcublas.cublasDgemmStridedBatched(
        handle,
        _CUBLAS_OP[transa],
        _CUBLAS_OP[transb],
        m,
        n,
        k,
        ctypes.byref(ctypes.c_double(alpha)),
        int(A),
        lda,
        strideA,
        int(B),
        ldb,
        strideB,
        ctypes.byref(ctypes.c_double(beta)),
        int(C),
        ldc,
        strideC,
        batchCount,
    )
    cublasCheckStatus(status)

cublasDgemmStridedBatched.__doc__ = _GEMM_STRIDED_BATCHED_doc.substitute(
    precision="double precision",
    real="real",
    num_type="numpy.float64",
)

@_cublas_version_req(8.0)
def cublasCgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount):
    """Matrix-matrix multiplication of a batch of matrices."""

    assert _libcublas
    status = _libcublas.cublasCgemmStridedBatched(
        handle,
        _CUBLAS_OP[transa],
        _CUBLAS_OP[transb],
        m,
        n,
        k,
        ctypes.byref(cuda.cuFloatComplex(alpha.real, alpha.imag)),
        int(A),
        lda,
        strideA,
        int(B),
        ldb,
        strideB,
        ctypes.byref(cuda.cuFloatComplex(beta.real, beta.imag)),
        int(C),
        ldc,
        strideC,
        batchCount,
    )
    cublasCheckStatus(status)

cublasCgemmStridedBatched.__doc__ = _GEMM_STRIDED_BATCHED_doc.substitute(
    precision="single precision",
    real="complex",
    num_type="numpy.complex64",
)

@_cublas_version_req(8.0)
def cublasZgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount):
    """Matrix-matrix multiplication of a batch of matrices."""

    assert _libcublas
    status = _libcublas.cublasZgemmStridedBatched(
        handle,
        _CUBLAS_OP[transa],
        _CUBLAS_OP[transb],
        m,
        n,
        k,
        ctypes.byref(cuda.cuDoubleComplex(alpha.real, alpha.imag)),
        int(A),
        lda,
        strideA,
        int(B),
        ldb,
        strideB,
        ctypes.byref(cuda.cuDoubleComplex(beta.real, beta.imag)),
        int(C),
        ldc,
        strideC,
        batchCount,
    )
    cublasCheckStatus(status)

cublasZgemmStridedBatched.__doc__ = _GEMM_STRIDED_BATCHED_doc.substitute(
    precision="double precision",
    real="complex",
    num_type="numpy.complex128",
)

if __name__ == "__main__":
    import doctest

    doctest.testmod()
