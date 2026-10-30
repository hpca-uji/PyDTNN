#!/usr/bin/env python
"""Python interface to CUDA driver functions."""

# Source: https://github.com/lebedov/scikit-cuda

import ctypes

from pydtnn.utils.libs import load_library

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


# Load library:
_libcuda = load_library("cuda")


# Exceptions corresponding to various CUDA driver errors:
class CUDA_ERROR(Exception):
    """CUDA error."""


class CUDA_ERROR_INVALID_VALUE(CUDA_ERROR):
    """CUDA error: Invalid value."""


class CUDA_ERROR_OUT_OF_MEMORY(CUDA_ERROR):
    """CUDA error: Out of memory."""


class CUDA_ERROR_NOT_INITIALIZED(CUDA_ERROR):
    """CUDA error: Not initialized."""


class CUDA_ERROR_DEINITIALIZED(CUDA_ERROR):
    """CUDA error: Deinitialized."""


class CUDA_ERROR_PROFILER_DISABLED(CUDA_ERROR):
    """CUDA error: Profiler disabled."""


class CUDA_ERROR_PROFILER_NOT_INITIALIZED(CUDA_ERROR):
    """CUDA error: Profiler not initialized."""


class CUDA_ERROR_PROFILER_ALREADY_STARTED(CUDA_ERROR):
    """CUDA error: Profiler already started."""


class CUDA_ERROR_PROFILER_ALREADY_STOPPED(CUDA_ERROR):
    """CUDA error: Profiler already stopped."""


class CUDA_ERROR_NO_DEVICE(CUDA_ERROR):
    """CUDA error: No device."""


class CUDA_ERROR_INVALID_DEVICE(CUDA_ERROR):
    """CUDA error: Invalid device."""


class CUDA_ERROR_INVALID_IMAGE(CUDA_ERROR):
    """CUDA error: Invalid image."""


class CUDA_ERROR_INVALID_CONTEXT(CUDA_ERROR):
    """CUDA error: Invalid context."""


class CUDA_ERROR_CONTEXT_ALREADY_CURRENT(CUDA_ERROR):
    """CUDA error: Context already current."""


class CUDA_ERROR_MAP_FAILED(CUDA_ERROR):
    """CUDA error: Map failed."""


class CUDA_ERROR_UNMAP_FAILED(CUDA_ERROR):
    """CUDA error: Unmap failed."""


class CUDA_ERROR_ARRAY_IS_MAPPED(CUDA_ERROR):
    """CUDA error: Array is mapped."""


class CUDA_ERROR_ALREADY_MAPPED(CUDA_ERROR):
    """CUDA error: Already mapped."""


class CUDA_ERROR_NO_BINARY_FOR_GPU(CUDA_ERROR):
    """CUDA error: No binary for GPU."""


class CUDA_ERROR_ALREADY_ACQUIRED(CUDA_ERROR):
    """CUDA error: Already acquired."""


class CUDA_ERROR_NOT_MAPPED(CUDA_ERROR):
    """CUDA error: Not mapped."""


class CUDA_ERROR_NOT_MAPPED_AS_ARRAY(CUDA_ERROR):
    """CUDA error: Not mapped as array."""


class CUDA_ERROR_NOT_MAPPED_AS_POINTER(CUDA_ERROR):
    """CUDA error: Not mapped as pointer."""


class CUDA_ERROR_ECC_UNCORRECTABLE(CUDA_ERROR):
    """CUDA error: ECC uncorrectable."""


class CUDA_ERROR_UNSUPPORTED_LIMIT(CUDA_ERROR):
    """CUDA error: Unsupported limit."""


class CUDA_ERROR_CONTEXT_ALREADY_IN_USE(CUDA_ERROR):
    """CUDA error: Context already in use."""


class CUDA_ERROR_PEER_ACCESS_UNSUPPORTED(CUDA_ERROR):
    """CUDA error: Peer access unsupported."""


class CUDA_ERROR_INVALID_PTX(CUDA_ERROR):
    """CUDA error: Invalid PTX."""


class CUDA_ERROR_INVALID_GRAPHICS_CONTEXT(CUDA_ERROR):
    """CUDA error: Invalid graphics context."""


class CUDA_ERROR_INVALID_SOURCE(CUDA_ERROR):
    """CUDA error: Invalid source."""


class CUDA_ERROR_FILE_NOT_FOUND(CUDA_ERROR):
    """CUDA error: File not found."""


class CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND(CUDA_ERROR):
    """CUDA error: Shared object symbol not found."""


class CUDA_ERROR_SHARED_OBJECT_INIT_FAILED(CUDA_ERROR):
    """CUDA error: Shared object init failed."""


class CUDA_ERROR_OPERATING_SYSTEM(CUDA_ERROR):
    """CUDA error: Operating system."""


class CUDA_ERROR_INVALID_HANDLE(CUDA_ERROR):
    """CUDA error: Invalid handle."""


class CUDA_ERROR_NOT_FOUND(CUDA_ERROR):
    """CUDA error: Not found."""


class CUDA_ERROR_NOT_READY(CUDA_ERROR):
    """CUDA error: Not ready."""


class CUDA_ERROR_ILLEGAL_ADDRESS(CUDA_ERROR):
    """CUDA error: Illegal address."""


class CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES(CUDA_ERROR):
    """CUDA error: Launch out of resources."""


class CUDA_ERROR_LAUNCH_TIMEOUT(CUDA_ERROR):
    """CUDA error: Launch timeout."""


class CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING(CUDA_ERROR):
    """CUDA error: Launch incompatible texturing."""


class CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED(CUDA_ERROR):
    """CUDA error: Peer access already enabled."""


class CUDA_ERROR_PEER_ACCESS_NOT_ENABLED(CUDA_ERROR):
    """CUDA error: Peer access not enabled."""


class CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE(CUDA_ERROR):
    """CUDA error: Primary context active."""


class CUDA_ERROR_CONTEXT_IS_DESTROYED(CUDA_ERROR):
    """CUDA error: Context is destroyed."""


class CUDA_ERROR_ASSERT(CUDA_ERROR):
    """CUDA error: Assert."""


class CUDA_ERROR_TOO_MANY_PEERS(CUDA_ERROR):
    """CUDA error: Too many peers."""


class CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED(CUDA_ERROR):
    """CUDA error: Host memory already registered."""


class CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED(CUDA_ERROR):
    """CUDA error: Host memory not registered."""


class CUDA_ERROR_HARDWARE_STACK_ERROR(CUDA_ERROR):
    """CUDA error: Hardware stack error."""


class CUDA_ERROR_ILLEGAL_INSTRUCTION(CUDA_ERROR):
    """CUDA error: Illegal instruction."""


class CUDA_ERROR_MISALIGNED_ADDRESS(CUDA_ERROR):
    """CUDA error: Misaligned address."""


class CUDA_ERROR_INVALID_ADDRESS_SPACE(CUDA_ERROR):
    """CUDA error: Invalid address space."""


class CUDA_ERROR_INVALID_PC(CUDA_ERROR):
    """CUDA error: Invalid PC."""


class CUDA_ERROR_LAUNCH_FAILED(CUDA_ERROR):
    """CUDA error: Launch failed."""


class CUDA_ERROR_NOT_PERMITTED(CUDA_ERROR):
    """CUDA error: Not permitted."""


class CUDA_ERROR_NOT_SUPPORTED(CUDA_ERROR):
    """CUDA error: Not supported."""


class CUDA_ERROR_UNKNOWN(CUDA_ERROR):
    """CUDA error: Unknown."""


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


def cuCheckStatus(status: int) -> None:
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


def cuPointerGetAttribute(attribute: int, ptr: int) -> ctypes.c_void_p:
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
