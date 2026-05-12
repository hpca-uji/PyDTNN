#!/usr/bin/env python
# Source: https://github.com/lebedov/scikit-cuda

"""
Python interface to CUDA driver functions.
"""

import ctypes
import sys

# Load library:
__all__ = (
    "CUDA_ERROR",
    "CUDA_ERROR_ALREADY_ACQUIRED",
    "CUDA_ERROR_ALREADY_MAPPED",
    "CUDA_ERROR_ARRAY_IS_MAPPED",
    "CUDA_ERROR_ASSERT",
    "CUDA_ERROR_CONTEXT_ALREADY_CURRENT",
    "CUDA_ERROR_CONTEXT_ALREADY_IN_USE",
    "CUDA_ERROR_CONTEXT_IS_DESTROYED",
    "CUDA_ERROR_DEINITIALIZED",
    "CUDA_ERROR_ECC_UNCORRECTABLE",
    "CUDA_ERROR_FILE_NOT_FOUND",
    "CUDA_ERROR_HARDWARE_STACK_ERROR",
    "CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED",
    "CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED",
    "CUDA_ERROR_ILLEGAL_ADDRESS",
    "CUDA_ERROR_ILLEGAL_INSTRUCTION",
    "CUDA_ERROR_INVALID_ADDRESS_SPACE",
    "CUDA_ERROR_INVALID_CONTEXT",
    "CUDA_ERROR_INVALID_DEVICE",
    "CUDA_ERROR_INVALID_GRAPHICS_CONTEXT",
    "CUDA_ERROR_INVALID_HANDLE",
    "CUDA_ERROR_INVALID_IMAGE",
    "CUDA_ERROR_INVALID_PC",
    "CUDA_ERROR_INVALID_PTX",
    "CUDA_ERROR_INVALID_SOURCE",
    "CUDA_ERROR_INVALID_VALUE",
    "CUDA_ERROR_LAUNCH_FAILED",
    "CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING",
    "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES",
    "CUDA_ERROR_LAUNCH_TIMEOUT",
    "CUDA_ERROR_MAP_FAILED",
    "CUDA_ERROR_MISALIGNED_ADDRESS",
    "CUDA_ERROR_NOT_FOUND",
    "CUDA_ERROR_NOT_INITIALIZED",
    "CUDA_ERROR_NOT_MAPPED",
    "CUDA_ERROR_NOT_MAPPED_AS_ARRAY",
    "CUDA_ERROR_NOT_MAPPED_AS_POINTER",
    "CUDA_ERROR_NOT_PERMITTED",
    "CUDA_ERROR_NOT_READY",
    "CUDA_ERROR_NOT_SUPPORTED",
    "CUDA_ERROR_NO_BINARY_FOR_GPU",
    "CUDA_ERROR_NO_DEVICE",
    "CUDA_ERROR_OPERATING_SYSTEM",
    "CUDA_ERROR_OUT_OF_MEMORY",
    "CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED",
    "CUDA_ERROR_PEER_ACCESS_NOT_ENABLED",
    "CUDA_ERROR_PEER_ACCESS_UNSUPPORTED",
    "CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE",
    "CUDA_ERROR_PROFILER_ALREADY_STARTED",
    "CUDA_ERROR_PROFILER_ALREADY_STOPPED",
    "CUDA_ERROR_PROFILER_DISABLED",
    "CUDA_ERROR_PROFILER_NOT_INITIALIZED",
    "CUDA_ERROR_SHARED_OBJECT_INIT_FAILED",
    "CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND",
    "CUDA_ERROR_TOO_MANY_PEERS",
    "CUDA_ERROR_UNKNOWN",
    "CUDA_ERROR_UNMAP_FAILED",
    "CUDA_ERROR_UNSUPPORTED_LIMIT",
    "cuCheckStatus",
    "cuPointerGetAttribute",
)

if "linux" in sys.platform:
    _libcuda_libname_list = ["libcuda.so"]
elif sys.platform == "darwin":
    _libcuda_libname_list = ["libcuda.dylib"]
elif sys.platform == "win32":
    _libcuda_libname_list = ["cuda.dll", "nvcuda.dll"]
else:
    raise RuntimeError("unsupported platform")

# Print understandable error message when library cannot be found:
_libcuda = None
for _libcuda_libname in _libcuda_libname_list:
    try:
        if sys.platform == "win32":
            _libcuda = ctypes.windll.LoadLibrary(_libcuda_libname)
        else:
            _libcuda = ctypes.cdll.LoadLibrary(_libcuda_libname)
    except OSError:
        pass
    else:
        break
if _libcuda is None:
    raise OSError("CUDA driver library not found")

# Exceptions corresponding to various CUDA driver errors:


class CUDA_ERROR(Exception):
    """CUDA error."""

    pass


class CUDA_ERROR_INVALID_VALUE(CUDA_ERROR):
    """CUDA error: Invalid value."""

    pass


class CUDA_ERROR_OUT_OF_MEMORY(CUDA_ERROR):
    """CUDA error: Out of memory."""

    pass


class CUDA_ERROR_NOT_INITIALIZED(CUDA_ERROR):
    """CUDA error: Not initialized."""

    pass


class CUDA_ERROR_DEINITIALIZED(CUDA_ERROR):
    """CUDA error: Deinitialized."""

    pass


class CUDA_ERROR_PROFILER_DISABLED(CUDA_ERROR):
    """CUDA error: Profiler disabled."""

    pass


class CUDA_ERROR_PROFILER_NOT_INITIALIZED(CUDA_ERROR):
    """CUDA error: Profiler not initialized."""

    pass


class CUDA_ERROR_PROFILER_ALREADY_STARTED(CUDA_ERROR):
    """CUDA error: Profiler already started."""

    pass


class CUDA_ERROR_PROFILER_ALREADY_STOPPED(CUDA_ERROR):
    """CUDA error: Profiler already stopped."""

    pass


class CUDA_ERROR_NO_DEVICE(CUDA_ERROR):
    """CUDA error: No device."""

    pass


class CUDA_ERROR_INVALID_DEVICE(CUDA_ERROR):
    """CUDA error: Invalid device."""

    pass


class CUDA_ERROR_INVALID_IMAGE(CUDA_ERROR):
    """CUDA error: Invalid image."""

    pass


class CUDA_ERROR_INVALID_CONTEXT(CUDA_ERROR):
    """CUDA error: Invalid context."""

    pass


class CUDA_ERROR_CONTEXT_ALREADY_CURRENT(CUDA_ERROR):
    """CUDA error: Context already current."""

    pass


class CUDA_ERROR_MAP_FAILED(CUDA_ERROR):
    """CUDA error: Map failed."""

    pass


class CUDA_ERROR_UNMAP_FAILED(CUDA_ERROR):
    """CUDA error: Unmap failed."""

    pass


class CUDA_ERROR_ARRAY_IS_MAPPED(CUDA_ERROR):
    """CUDA error: Array is mapped."""

    pass


class CUDA_ERROR_ALREADY_MAPPED(CUDA_ERROR):
    """CUDA error: Already mapped."""

    pass


class CUDA_ERROR_NO_BINARY_FOR_GPU(CUDA_ERROR):
    """CUDA error: No binary for GPU."""

    pass


class CUDA_ERROR_ALREADY_ACQUIRED(CUDA_ERROR):
    """CUDA error: Already acquired."""

    pass


class CUDA_ERROR_NOT_MAPPED(CUDA_ERROR):
    """CUDA error: Not mapped."""

    pass


class CUDA_ERROR_NOT_MAPPED_AS_ARRAY(CUDA_ERROR):
    """CUDA error: Not mapped as array."""

    pass


class CUDA_ERROR_NOT_MAPPED_AS_POINTER(CUDA_ERROR):
    """CUDA error: Not mapped as pointer."""

    pass


class CUDA_ERROR_ECC_UNCORRECTABLE(CUDA_ERROR):
    """CUDA error: ECC uncorrectable."""

    pass


class CUDA_ERROR_UNSUPPORTED_LIMIT(CUDA_ERROR):
    """CUDA error: Unsupported limit."""

    pass


class CUDA_ERROR_CONTEXT_ALREADY_IN_USE(CUDA_ERROR):
    """CUDA error: Context already in use."""

    pass


class CUDA_ERROR_PEER_ACCESS_UNSUPPORTED(CUDA_ERROR):
    """CUDA error: Peer access unsupported."""

    pass


class CUDA_ERROR_INVALID_PTX(CUDA_ERROR):
    """CUDA error: Invalid PTX."""

    pass


class CUDA_ERROR_INVALID_GRAPHICS_CONTEXT(CUDA_ERROR):
    """CUDA error: Invalid graphics context."""

    pass


class CUDA_ERROR_INVALID_SOURCE(CUDA_ERROR):
    """CUDA error: Invalid source."""

    pass


class CUDA_ERROR_FILE_NOT_FOUND(CUDA_ERROR):
    """CUDA error: File not found."""

    pass


class CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND(CUDA_ERROR):
    """CUDA error: Shared object symbol not found."""

    pass


class CUDA_ERROR_SHARED_OBJECT_INIT_FAILED(CUDA_ERROR):
    """CUDA error: Shared object init failed."""

    pass


class CUDA_ERROR_OPERATING_SYSTEM(CUDA_ERROR):
    """CUDA error: Operating system."""

    pass


class CUDA_ERROR_INVALID_HANDLE(CUDA_ERROR):
    """CUDA error: Invalid handle."""

    pass


class CUDA_ERROR_NOT_FOUND(CUDA_ERROR):
    """CUDA error: Not found."""

    pass


class CUDA_ERROR_NOT_READY(CUDA_ERROR):
    """CUDA error: Not ready."""

    pass


class CUDA_ERROR_ILLEGAL_ADDRESS(CUDA_ERROR):
    """CUDA error: Illegal address."""

    pass


class CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES(CUDA_ERROR):
    """CUDA error: Launch out of resources."""

    pass


class CUDA_ERROR_LAUNCH_TIMEOUT(CUDA_ERROR):
    """CUDA error: Launch timeout."""

    pass


class CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING(CUDA_ERROR):
    """CUDA error: Launch incompatible texturing."""

    pass


class CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED(CUDA_ERROR):
    """CUDA error: Peer access already enabled."""

    pass


class CUDA_ERROR_PEER_ACCESS_NOT_ENABLED(CUDA_ERROR):
    """CUDA error: Peer access not enabled."""

    pass


class CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE(CUDA_ERROR):
    """CUDA error: Primary context active."""

    pass


class CUDA_ERROR_CONTEXT_IS_DESTROYED(CUDA_ERROR):
    """CUDA error: Context is destroyed."""

    pass


class CUDA_ERROR_ASSERT(CUDA_ERROR):
    """CUDA error: Assert."""

    pass


class CUDA_ERROR_TOO_MANY_PEERS(CUDA_ERROR):
    """CUDA error: Too many peers."""

    pass


class CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED(CUDA_ERROR):
    """CUDA error: Host memory already registered."""

    pass


class CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED(CUDA_ERROR):
    """CUDA error: Host memory not registered."""

    pass


class CUDA_ERROR_HARDWARE_STACK_ERROR(CUDA_ERROR):
    """CUDA error: Hardware stack error."""

    pass


class CUDA_ERROR_ILLEGAL_INSTRUCTION(CUDA_ERROR):
    """CUDA error: Illegal instruction."""

    pass


class CUDA_ERROR_MISALIGNED_ADDRESS(CUDA_ERROR):
    """CUDA error: Misaligned address."""

    pass


class CUDA_ERROR_INVALID_ADDRESS_SPACE(CUDA_ERROR):
    """CUDA error: Invalid address space."""

    pass


class CUDA_ERROR_INVALID_PC(CUDA_ERROR):
    """CUDA error: Invalid PC."""

    pass


class CUDA_ERROR_LAUNCH_FAILED(CUDA_ERROR):
    """CUDA error: Launch failed."""

    pass


class CUDA_ERROR_NOT_PERMITTED(CUDA_ERROR):
    """CUDA error: Not permitted."""

    pass


class CUDA_ERROR_NOT_SUPPORTED(CUDA_ERROR):
    """CUDA error: Not supported."""

    pass


class CUDA_ERROR_UNKNOWN(CUDA_ERROR):
    """CUDA error: Unknown."""

    pass


CUDA_EXCEPTIONS = {
    1: CUDA_ERROR_INVALID_VALUE,
    2: CUDA_ERROR_OUT_OF_MEMORY,
    3: CUDA_ERROR_NOT_INITIALIZED,
    4: CUDA_ERROR_DEINITIALIZED,
    5: CUDA_ERROR_PROFILER_DISABLED,
    6: CUDA_ERROR_PROFILER_NOT_INITIALIZED,
    7: CUDA_ERROR_PROFILER_ALREADY_STARTED,
    8: CUDA_ERROR_PROFILER_ALREADY_STOPPED,
    100: CUDA_ERROR_NO_DEVICE,
    101: CUDA_ERROR_INVALID_DEVICE,
    200: CUDA_ERROR_INVALID_IMAGE,
    201: CUDA_ERROR_INVALID_CONTEXT,
    202: CUDA_ERROR_CONTEXT_ALREADY_CURRENT,
    205: CUDA_ERROR_MAP_FAILED,
    206: CUDA_ERROR_UNMAP_FAILED,
    207: CUDA_ERROR_ARRAY_IS_MAPPED,
    208: CUDA_ERROR_ALREADY_MAPPED,
    209: CUDA_ERROR_NO_BINARY_FOR_GPU,
    210: CUDA_ERROR_ALREADY_ACQUIRED,
    211: CUDA_ERROR_NOT_MAPPED,
    212: CUDA_ERROR_NOT_MAPPED_AS_ARRAY,
    213: CUDA_ERROR_NOT_MAPPED_AS_POINTER,
    214: CUDA_ERROR_ECC_UNCORRECTABLE,
    215: CUDA_ERROR_UNSUPPORTED_LIMIT,
    216: CUDA_ERROR_CONTEXT_ALREADY_IN_USE,
    217: CUDA_ERROR_PEER_ACCESS_UNSUPPORTED,
    218: CUDA_ERROR_INVALID_PTX,
    219: CUDA_ERROR_INVALID_GRAPHICS_CONTEXT,
    300: CUDA_ERROR_INVALID_SOURCE,
    301: CUDA_ERROR_FILE_NOT_FOUND,
    302: CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND,
    303: CUDA_ERROR_SHARED_OBJECT_INIT_FAILED,
    304: CUDA_ERROR_OPERATING_SYSTEM,
    400: CUDA_ERROR_INVALID_HANDLE,
    500: CUDA_ERROR_NOT_FOUND,
    600: CUDA_ERROR_NOT_READY,
    700: CUDA_ERROR_ILLEGAL_ADDRESS,
    701: CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES,
    702: CUDA_ERROR_LAUNCH_TIMEOUT,
    703: CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING,
    704: CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED,
    705: CUDA_ERROR_PEER_ACCESS_NOT_ENABLED,
    708: CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE,
    709: CUDA_ERROR_CONTEXT_IS_DESTROYED,
    710: CUDA_ERROR_ASSERT,
    711: CUDA_ERROR_TOO_MANY_PEERS,
    712: CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED,
    713: CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED,
    714: CUDA_ERROR_HARDWARE_STACK_ERROR,
    715: CUDA_ERROR_ILLEGAL_INSTRUCTION,
    716: CUDA_ERROR_MISALIGNED_ADDRESS,
    717: CUDA_ERROR_INVALID_ADDRESS_SPACE,
    718: CUDA_ERROR_INVALID_PC,
    719: CUDA_ERROR_LAUNCH_FAILED,
    800: CUDA_ERROR_NOT_PERMITTED,
    801: CUDA_ERROR_NOT_SUPPORTED,
    999: CUDA_ERROR_UNKNOWN,
}


def cuCheckStatus(status):
    """
    Raise CUDA exception.

    Raise an exception corresponding to the specified CUDA driver
    error code.

    Parameters
    ----------
    status : int
        CUDA driver error code.

    See Also
    --------
    CUDA_EXCEPTIONS
    """

    if status != 0:
        try:
            e = CUDA_EXCEPTIONS[status]
        except KeyError:
            raise CUDA_ERROR
        else:
            raise e


CU_POINTER_ATTRIBUTE_CONTEXT = 1
CU_POINTER_ATTRIBUTE_MEMORY_TYPE = 2
CU_POINTER_ATTRIBUTE_DEVICE_POINTER = 3
CU_POINTER_ATTRIBUTE_HOST_POINTER = 4

_libcuda.cuPointerGetAttribute.restype = int
_libcuda.cuPointerGetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint]


def cuPointerGetAttribute(attribute, ptr):
    """
    Get a pointer attribute.

    Retrieves a specific attribute of a CUDA pointer.

    Parameters
    ----------
    attribute : int
        The attribute to query (e.g., CU_POINTER_ATTRIBUTE_CONTEXT).
    ptr : int
        The pointer to query.

    Returns
    -------
    ctypes.c_void_p
        The value of the requested attribute.

    Raises
    ------
    CUDA_ERROR
        If the CUDA driver call fails.
    """
    data = ctypes.c_void_p()
    assert _libcuda
    status = _libcuda.cuPointerGetAttribute(data, attribute, ptr)
    cuCheckStatus(status)
    return data