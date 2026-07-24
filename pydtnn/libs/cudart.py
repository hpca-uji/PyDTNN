#!/usr/bin/env python
"""Python interface to CUDA runtime functions."""

from __future__ import annotations

import ctypes
import functools
import re
import sys
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

# Source: https://github.com/lebedov/scikit-cuda


if TYPE_CHECKING:
    import gpuarray


# Load library:
__all__ = (
    "POINTER",
    "cuDoubleComplex",
    "cuFloatComplex",
    "cudaCheckStatus",
    "cudaDriverGetVersion",
    "cudaError",
    "cudaErrorAssert",
    "cudaErrorCudartUnloading",
    "cudaErrorDeviceAlreadyInUse",
    "cudaErrorDevicesUnavailable",
    "cudaErrorDuplicateSurfaceName",
    "cudaErrorDuplicateTextureName",
    "cudaErrorDuplicateVariableName",
    "cudaErrorECCUncorrectable",
    "cudaErrorHardwareStackError",
    "cudaErrorHostMemoryAlreadyRegistered",
    "cudaErrorHostMemoryNotRegistered",
    "cudaErrorIllegalAddress",
    "cudaErrorIllegalInstruction",
    "cudaErrorIncompatibleDriverContext",
    "cudaErrorInitializationError",
    "cudaErrorInsufficientDriver",
    "cudaErrorInvalidAddressSpace",
    "cudaErrorInvalidChannelDescriptor",
    "cudaErrorInvalidConfiguration",
    "cudaErrorInvalidDevice",
    "cudaErrorInvalidDeviceFunction",
    "cudaErrorInvalidDevicePointer",
    "cudaErrorInvalidFilterSetting",
    "cudaErrorInvalidGraphicsContext",
    "cudaErrorInvalidHostPointer",
    "cudaErrorInvalidKernelImage",
    "cudaErrorInvalidMemcpyDirection",
    "cudaErrorInvalidNormSetting",
    "cudaErrorInvalidPc",
    "cudaErrorInvalidPitchValue",
    "cudaErrorInvalidPtx",
    "cudaErrorInvalidResourceHandle",
    "cudaErrorInvalidSurface",
    "cudaErrorInvalidSymbol",
    "cudaErrorInvalidTexture",
    "cudaErrorInvalidTextureBinding",
    "cudaErrorInvalidValue",
    "cudaErrorLaunchFailure",
    "cudaErrorLaunchFileScopedSurf",
    "cudaErrorLaunchFileScopedTex",
    "cudaErrorLaunchMaxDepthExceeded",
    "cudaErrorLaunchOutOfResources",
    "cudaErrorLaunchPendingCountExceeded",
    "cudaErrorLaunchTimeout",
    "cudaErrorMapBufferObjectFailed",
    "cudaErrorMemoryAllocation",
    "cudaErrorMemoryValueTooLarge",
    "cudaErrorMisalignedAddress",
    "cudaErrorMissingConfiguration",
    "cudaErrorMixedDeviceExecution",
    "cudaErrorNoDevice",
    "cudaErrorNoKernelImageForDevice",
    "cudaErrorNotPermitted",
    "cudaErrorNotReady",
    "cudaErrorNotSupported",
    "cudaErrorNotYetImplemented",
    "cudaErrorOperatingSystem",
    "cudaErrorPeerAccessAlreadyEnabled",
    "cudaErrorPeerAccessNotEnabled",
    "cudaErrorPeerAccessUnsupported",
    "cudaErrorPriorLaunchFailure",
    "cudaErrorProfilerAlreadyStarted",
    "cudaErrorProfilerAlreadyStopped",
    "cudaErrorProfilerDisabled",
    "cudaErrorProfilerNotInitialized",
    "cudaErrorSetOnActiveProcess",
    "cudaErrorSharedObjectInitFailed",
    "cudaErrorSharedObjectSymbolNotFound",
    "cudaErrorStartupFailure",
    "cudaErrorSyncDepthExceeded",
    "cudaErrorSynchronizationError",
    "cudaErrorTextureFetchFailed",
    "cudaErrorTextureNotBound",
    "cudaErrorTooManyPeers",
    "cudaErrorUnknown",
    "cudaErrorUnmapBufferObjectFailed",
    "cudaErrorUnsupportedLimit",
    "cudaFree",
    "cudaGetDevice",
    "cudaGetErrorString",
    "cudaMalloc",
    "cudaMallocPitch",
    "cudaMemGetInfo",
    "cudaMemcpy_dtoh",
    "cudaMemcpy_htod",
    "cudaPointerAttributes",
    "cudaPointerGetAttributes",
    "cudaRuntimeGetVersion",
    "cudaSetDevice",
    "double2",
    "float2",
    "gpuarray_ptr",
)

_linux_version_list = [
    11.0,
    10.2,
    10.1,
    10.0,
    9.2,
    9.1,
    9.0,
    8.0,
    7.5,
    7.0,
    6.5,
    6.0,
    5.5,
    5.0,
    4.0,
]
_win32_version_list = [110, 102, 101, 100, 92, 91, 90, 80, 75, 70, 65, 60, 55, 50, 40]
if "linux" in sys.platform:
    _libcudart_libname_list = ["libcudart.so"] + [
        "libcudart.so.%s" % v for v in _linux_version_list
    ]
elif sys.platform == "darwin":
    _libcudart_libname_list = ["libcudart.dylib"]
elif sys.platform == "win32":
    if sys.maxsize > 2**32:
        _libcudart_libname_list = ["cudart.dll"] + [
            "cudart64_%s.dll" % v for v in _win32_version_list
        ]
    else:
        _libcudart_libname_list = ["cudart.dll"] + [
            "cudart32_%s.dll" % v for v in _win32_version_list
        ]
else:
    raise RuntimeError("unsupported platform")

# Print understandable error message when library cannot be found:
_libcudart = None
for _libcudart_libname in _libcudart_libname_list:
    try:
        if sys.platform == "win32":
            _libcudart = ctypes.windll.LoadLibrary(_libcudart_libname)
        else:
            _libcudart = ctypes.cdll.LoadLibrary(_libcudart_libname)
    except OSError:
        pass
    else:
        break
if _libcudart is None:
    raise OSError("CUDA runtime library not found")

# Code adapted from PARRET:


@functools.wraps(ctypes.POINTER)
def POINTER(type: type) -> type[ctypes._Pointer]:
    """
    Create ctypes pointer to object.

    Notes
    -----
    This function converts None to a real NULL pointer because of bug
    in how ctypes handles None on 64-bit platforms.

    """

    p = ctypes.POINTER(type)
    if not isinstance(p.from_param, classmethod):

        def from_param[T](cls: type[T], x: T) -> T:
            if x is None:
                return cls()
            else:
                return x

        p.from_param = classmethod(from_param)  # pyright: ignore[reportAttributeAccessIssue]

    return p


# Classes corresponding to CUDA vector structures:


class float2(ctypes.Structure):
    """CUDA float2 structure."""

    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float)]


class cuFloatComplex(float2):
    """CUDA cuFloatComplex structure."""

    @property
    def value(self) -> complex:
        """Return the complex representation of the structure."""
        return complex(self.x, self.y)


class double2(ctypes.Structure):
    """CUDA double2 structure."""

    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]


class cuDoubleComplex(double2):
    """CUDA cuDoubleComplex structure."""

    @property
    def value(self) -> complex:
        """Return the complex representation of the structure."""
        return complex(self.x, self.y)


def gpuarray_ptr(g: gpuarray.GPUArray) -> ctypes.c_void_p:
    """
    Return ctypes pointer to data in GPUAarray object.

    Parameters
    ----------
    g : gpuarray.GPUArray
        GPUArray object.

    Returns
    -------
    ptr : ctypes pointer
        Pointer to the GPUArray's data.

    Raises
    ------
    ValueError
        If the GPUArray's dtype is not recognized.
    """

    addr = int(g.gpudata)
    if g.dtype == np.int8:
        return ctypes.cast(addr, POINTER(ctypes.c_byte))  # pyright: ignore[reportReturnType]
    if g.dtype == np.uint8:
        return ctypes.cast(addr, POINTER(ctypes.c_ubyte))  # pyright: ignore[reportReturnType]
    if g.dtype == np.int16:
        return ctypes.cast(addr, POINTER(ctypes.c_short))  # pyright: ignore[reportReturnType]
    if g.dtype == np.uint16:
        return ctypes.cast(addr, POINTER(ctypes.c_ushort))  # pyright: ignore[reportReturnType]
    if g.dtype == np.int32:
        return ctypes.cast(addr, POINTER(ctypes.c_int))  # pyright: ignore[reportReturnType]
    if g.dtype == np.uint32:
        return ctypes.cast(addr, POINTER(ctypes.c_uint))  # pyright: ignore[reportReturnType]
    if g.dtype == np.int64:
        return ctypes.cast(addr, POINTER(ctypes.c_long))  # pyright: ignore[reportReturnType]
    if g.dtype == np.uint64:
        return ctypes.cast(addr, POINTER(ctypes.c_ulong))  # pyright: ignore[reportReturnType]
    if g.dtype == np.float32:
        return ctypes.cast(addr, POINTER(ctypes.c_float))  # pyright: ignore[reportReturnType]
    elif g.dtype == np.float64:
        return ctypes.cast(addr, POINTER(ctypes.c_double))  # pyright: ignore[reportReturnType]
    elif g.dtype == np.complex64:
        return ctypes.cast(addr, POINTER(cuFloatComplex))  # pyright: ignore[reportReturnType]
    elif g.dtype == np.complex128:
        return ctypes.cast(addr, POINTER(cuDoubleComplex))  # pyright: ignore[reportReturnType]
    else:
        raise ValueError("unrecognized type")


_libcudart.cudaGetErrorString.restype = ctypes.c_char_p
_libcudart.cudaGetErrorString.argtypes = [ctypes.c_int]


def cudaGetErrorString(e: int) -> str:
    """
    Retrieve CUDA error string.

    Return the string associated with the specified CUDA error status
    code.

    Parameters
    ----------
    e : int
        Error number.

    Returns
    -------
    s : str
        Error string.

    """

    assert _libcudart
    return _libcudart.cudaGetErrorString(e)


# Generic CUDA error:


class cudaError(Exception):
    """Base class for CUDA runtime errors."""

    pass


# Exceptions corresponding to various CUDA runtime errors:


class cudaErrorMissingConfiguration(cudaError):
    """
    cudaErrorMissingConfiguration

    The device function being invoked (usually via
    cudaLaunchKernel()) was not previously configured via the
    cudaConfigureCall() function.
    """

    __doc__ = _libcudart.cudaGetErrorString(1)


class cudaErrorMemoryAllocation(cudaError):
    """
    cudaErrorMemoryAllocation

    The API call failed because it was unable to allocate enough
    memory or other resources to perform the requested operation.
    """

    __doc__ = _libcudart.cudaGetErrorString(2)


class cudaErrorInitializationError(cudaError):
    """
    cudaErrorInitializationError

    The API call failed because the CUDA driver and runtime could
    not be initialized.
    """

    __doc__ = _libcudart.cudaGetErrorString(3)


class cudaErrorLaunchFailure(cudaError):
    """
    cudaErrorLaunchFailure

    An exception occurred on the device while executing a kernel.
    Common causes include dereferencing an invalid device pointer
    and accessing out of bounds shared memory. Less common cases
    can be system specific - more information about these cases
    can be found in the system specific user guide. This leaves
    the process in an inconsistent state and any further CUDA work
    will return the same error. To continue using CUDA, the process
    must be terminated and relaunched.
    """

    __doc__ = _libcudart.cudaGetErrorString(4)


class cudaErrorPriorLaunchFailure(cudaError):
    """cudaErrorPriorLaunchFailure"""

    __doc__ = _libcudart.cudaGetErrorString(5)


class cudaErrorLaunchTimeout(cudaError):
    """
    cudaErrorLaunchTimeout

    This indicates that the device kernel took too long to execute.
    This can only occur if timeouts are enabled - see the device attribute
    cudaDevAttrKernelExecTimeout for more information. This leaves the
    process in an inconsistent state and any further CUDA work will
    return the same error. To continue using CUDA, the process must
    be terminated and relaunched.
    """

    __doc__ = _libcudart.cudaGetErrorString(6)


class cudaErrorLaunchOutOfResources(cudaError):
    """
    cudaErrorLaunchOutOfResources

    This indicates that a launch did not occur because it did not have
    appropriate resources. Although this error is similar to
    cudaErrorInvalidConfiguration, this error usually indicates that
    the user has attempted to pass too many arguments to the device kernel,
    or the kernel launch specifies too many threads for the kernel's
    register count.
    """

    __doc__ = _libcudart.cudaGetErrorString(7)


class cudaErrorInvalidDeviceFunction(cudaError):
    """
    cudaErrorInvalidDeviceFunction

    The requested device function does not exist or is not compiled for
    the proper device architecture.
    """

    __doc__ = _libcudart.cudaGetErrorString(8)


class cudaErrorInvalidConfiguration(cudaError):
    """
    cudaErrorInvalidConfiguration

    This indicates that a kernel launch is requesting resources that can
    never be satisfied by the current device. Requesting more shared memory
    per block than the device supports will trigger this error, as will
    requesting too many threads or blocks. See cudaDeviceProp for more
    device limitations.
    """

    __doc__ = _libcudart.cudaGetErrorString(9)


class cudaErrorInvalidDevice(cudaError):
    """
    cudaErrorInvalidDevice

    This indicates that the device ordinal supplied by the user does not
    correspond to a valid CUDA device or that the action requested is
    invalid for the specified device.
    """

    __doc__ = _libcudart.cudaGetErrorString(10)


class cudaErrorInvalidValue(cudaError):
    """
    cudaErrorInvalidValue

    This indicates that one or more of the parameters passed to the API
    call is not within an acceptable range of values.
    """

    __doc__ = _libcudart.cudaGetErrorString(11)


class cudaErrorInvalidPitchValue(cudaError):
    """
    cudaErrorInvalidPitchValue

    This indicates that one or more of the pitch-related parameters passed
    to the API call is not within the acceptable range for pitch.
    """

    __doc__ = _libcudart.cudaGetErrorString(12)


class cudaErrorInvalidSymbol(cudaError):
    """
    cudaErrorInvalidSymbol

    This indicates that the symbol name/identifier passed to the API call
    is not a valid name or identifier.
    """

    __doc__ = _libcudart.cudaGetErrorString(13)


class cudaErrorMapBufferObjectFailed(cudaError):
    """
    cudaErrorMapBufferObjectFailed

    This indicates that the buffer object could not be mapped.
    """

    __doc__ = _libcudart.cudaGetErrorString(14)


class cudaErrorUnmapBufferObjectFailed(cudaError):
    """
    cudaErrorUnmapBufferObjectFailed

    This indicates that the buffer object could not be unmapped.
    """

    __doc__ = _libcudart.cudaGetErrorString(15)


class cudaErrorInvalidHostPointer(cudaError):
    """cudaErrorInvalidHostPointer"""

    __doc__ = _libcudart.cudaGetErrorString(16)


class cudaErrorInvalidDevicePointer(cudaError):
    """cudaErrorInvalidDevicePointer"""

    __doc__ = _libcudart.cudaGetErrorString(17)


class cudaErrorInvalidTexture(cudaError):
    """
    cudaErrorInvalidTexture

    This indicates that the texture passed to the API call is not a valid
    texture.
    """

    __doc__ = _libcudart.cudaGetErrorString(18)


class cudaErrorInvalidTextureBinding(cudaError):
    """
    cudaErrorInvalidTextureBinding

    This indicates that the texture binding is not valid. This occurs if
    you call cudaGetTextureAlignmentOffset() with an unbound texture.
    """

    __doc__ = _libcudart.cudaGetErrorString(19)


class cudaErrorInvalidChannelDescriptor(cudaError):
    """
    cudaErrorInvalidChannelDescriptor

    This indicates that the channel descriptor passed to the API call is
    not valid. This occurs if the format is not one of the formats
    specified by cudaChannelFormatKind, or if one of the dimensions
    is invalid.
    """

    __doc__ = _libcudart.cudaGetErrorString(20)


class cudaErrorInvalidMemcpyDirection(cudaError):
    """
    cudaErrorInvalidMemcpyDirection

    This indicates that the direction of the memcpy passed to the API
    call is not one of the types specified by cudaMemcpyKind.
    """

    __doc__ = _libcudart.cudaGetErrorString(21)


class cudaErrorTextureFetchFailed(cudaError):
    """cudaErrorTextureFetchFailed"""

    __doc__ = _libcudart.cudaGetErrorString(23)


class cudaErrorTextureNotBound(cudaError):
    """cudaErrorTextureNotBound"""

    __doc__ = _libcudart.cudaGetErrorString(24)


class cudaErrorSynchronizationError(cudaError):
    """cudaErrorSynchronizationError"""

    __doc__ = _libcudart.cudaGetErrorString(25)


class cudaErrorInvalidFilterSetting(cudaError):
    """
    cudaErrorInvalidFilterSetting

    This indicates that a non-float texture was being accessed with
    linear filtering. This is not supported by CUDA.
    """

    __doc__ = _libcudart.cudaGetErrorString(26)


class cudaErrorInvalidNormSetting(cudaError):
    """
    cudaErrorInvalidNormSetting

    This indicates that an attempt was made to read an unsupported data
    type as a normalized float. This is not supported by CUDA.
    """

    __doc__ = _libcudart.cudaGetErrorString(27)


class cudaErrorMixedDeviceExecution(cudaError):
    """cudaErrorMixedDeviceExecution"""

    __doc__ = _libcudart.cudaGetErrorString(28)


class cudaErrorCudartUnloading(cudaError):
    """
    cudaErrorCudartUnloading

    This indicates that a CUDA Runtime API call cannot be executed
    because it is being called during process shut down, at a point
    in time after CUDA driver has been unloaded.
    """

    __doc__ = _libcudart.cudaGetErrorString(29)


class cudaErrorUnknown(cudaError):
    """
    cudaErrorUnknown

    This indicates that an unknown internal error has occurred.
    """

    __doc__ = _libcudart.cudaGetErrorString(30)


class cudaErrorNotYetImplemented(cudaError):
    """cudaErrorNotYetImplemented"""

    __doc__ = _libcudart.cudaGetErrorString(31)


class cudaErrorMemoryValueTooLarge(cudaError):
    """cudaErrorMemoryValueTooLarge"""

    __doc__ = _libcudart.cudaGetErrorString(32)


class cudaErrorInvalidResourceHandle(cudaError):
    """
    cudaErrorInvalidResourceHandle

    This indicates that a resource handle passed to the API call was
    not valid. Resource handles are opaque types like cudaStream_t
    and cudaEvent_t.
    """

    __doc__ = _libcudart.cudaGetErrorString(33)


class cudaErrorNotReady(cudaError):
    """
    cudaErrorNotReady

    This indicates that asynchronous operations issued previously have
    not completed yet. This result is not actually an error, but must
    be indicated differently than cudaSuccess (which indicates completion).
    Calls that may return this value include cudaEventQuery() and
    cudaStreamQuery().
    """

    __doc__ = _libcudart.cudaGetErrorString(34)


class cudaErrorInsufficientDriver(cudaError):
    """
    cudaErrorInsufficientDriver

    This indicates that the installed NVIDIA CUDA driver is older than
    the CUDA runtime library. This is not a supported configuration.
    Users should install an updated NVIDIA display driver to allow the
    application to run.
    """

    __doc__ = _libcudart.cudaGetErrorString(35)


class cudaErrorSetOnActiveProcess(cudaError):
    """
    cudaErrorSetOnActiveProcess

    This indicates that the user has called cudaSetValidDevices(),
    cudaSetDeviceFlags(), cudaD3D9SetDirect3DDevice(),
    cudaD3D10SetDirect3DDevice, cudaD3D11SetDirect3DDevice(),
    or cudaVDPAUSetVDPAUDevice() after initializing the CUDA runtime
    by calling non-device management operations (allocating memory and
    launching kernels are examples of non-device management operations).
    This error can also be returned if using runtime/driver interoperability
    and there is an existing CUcontext active on the host thread.
    """

    __doc__ = _libcudart.cudaGetErrorString(36)


class cudaErrorInvalidSurface(cudaError):
    """
    cudaErrorInvalidSurface

    This indicates that the surface passed to the API call is not
    a valid surface.
    """

    __doc__ = _libcudart.cudaGetErrorString(37)


class cudaErrorNoDevice(cudaError):
    """
    cudaErrorNoDevice

    This indicates that no CUDA-capable devices were detected by
    the installed CUDA driver.
    """

    __doc__ = _libcudart.cudaGetErrorString(38)


class cudaErrorECCUncorrectable(cudaError):
    """
    cudaErrorECCUncorrectable

    This indicates that an uncorrectable ECC error was detected
    during execution.
    """

    __doc__ = _libcudart.cudaGetErrorString(39)


class cudaErrorSharedObjectSymbolNotFound(cudaError):
    """
    cudaErrorSharedObjectSymbolNotFound

    This indicates that a link to a shared object failed to resolve.
    """

    __doc__ = _libcudart.cudaGetErrorString(40)


class cudaErrorSharedObjectInitFailed(cudaError):
    """
    cudaErrorSharedObjectInitFailed

    This indicates that initialization of a shared object failed.
    """

    __doc__ = _libcudart.cudaGetErrorString(41)


class cudaErrorUnsupportedLimit(cudaError):
    """
    cudaErrorUnsupportedLimit

    This indicates that the cudaLimit passed to the API call is
    not supported by the active device.
    """

    __doc__ = _libcudart.cudaGetErrorString(42)


class cudaErrorDuplicateVariableName(cudaError):
    """
    cudaErrorDuplicateVariableName

    This indicates that multiple global or constant variables
    (across separate CUDA source files in the application) share
    the same string name.
    """

    __doc__ = _libcudart.cudaGetErrorString(43)


class cudaErrorDuplicateTextureName(cudaError):
    """
    cudaErrorDuplicateTextureName

    This indicates that multiple textures (across separate CUDA
    source files in the application) share the same string name.
    """

    __doc__ = _libcudart.cudaGetErrorString(44)


class cudaErrorDuplicateSurfaceName(cudaError):
    """
    cudaErrorDuplicateSurfaceName

    This indicates that multiple surfaces (across separate CUDA
    source files in the application) share the same string name.
    """

    __doc__ = _libcudart.cudaGetErrorString(45)


class cudaErrorDevicesUnavailable(cudaError):
    """
    cudaErrorDevicesUnavailable

    This indicates that all CUDA devices are busy or unavailable
    at the current time. Devices are often busy/unavailable due to
    use of cudaComputeModeProhibited, cudaComputeModeExclusiveProcess,
    or when long running CUDA kernels have filled up the GPU and are
    blocking new work from starting. They can also be unavailable
    due to memory constraints on a device that already has active
    CUDA work being performed.
    """

    __doc__ = _libcudart.cudaGetErrorString(46)


class cudaErrorInvalidKernelImage(cudaError):
    """
    cudaErrorInvalidKernelImage

    This indicates that the device kernel image is invalid.
    """

    __doc__ = _libcudart.cudaGetErrorString(47)


class cudaErrorNoKernelImageForDevice(cudaError):
    """
    cudaErrorNoKernelImageForDevice

    This indicates that there is no kernel image available that is
    suitable for the device. This can occur when a user specifies
    code generation options for a particular CUDA source file that
    do not include the corresponding device configuration.
    """

    __doc__ = _libcudart.cudaGetErrorString(48)


class cudaErrorIncompatibleDriverContext(cudaError):
    """
    cudaErrorIncompatibleDriverContext

    This indicates that the current context is not compatible with
    this the CUDA Runtime. This can only occur if you are using CUDA
    Runtime/Driver interoperability and have created an existing Driver
    context using the driver API. The Driver context may be incompatible
    either because the Driver context was created using an older
    version of the API, because the Runtime API call expects a primary
    driver context and the Driver context is not primary, or because
    the Driver context has been destroyed. Please see Interactions with
    the CUDA Driver API" for more information.
    """

    __doc__ = _libcudart.cudaGetErrorString(49)


class cudaErrorPeerAccessAlreadyEnabled(cudaError):
    """
    cudaErrorPeerAccessAlreadyEnabled

    This error indicates that a call to cudaDeviceEnablePeerAccess() is
    trying to re-enable peer addressing on from a context which has
    already had peer addressing enabled.
    """

    __doc__ = _libcudart.cudaGetErrorString(50)


class cudaErrorPeerAccessNotEnabled(cudaError):
    """
    cudaErrorPeerAccessNotEnabled

    This error indicates that cudaDeviceDisablePeerAccess() is trying
    to disable peer addressing which has not been enabled yet via
    cudaDeviceEnablePeerAccess().
    """

    __doc__ = _libcudart.cudaGetErrorString(51)


class cudaErrorDeviceAlreadyInUse(cudaError):
    """
    cudaErrorDeviceAlreadyInUse

    This indicates that a call tried to access an exclusive-thread
    device that is already in use by a different thread.
    """

    __doc__ = _libcudart.cudaGetErrorString(54)


class cudaErrorProfilerDisabled(cudaError):
    """
    cudaErrorProfilerDisabled

    This indicates profiler is not initialized for this run. This can
    happen when the application is running with external profiling
    tools like visual profiler.
    """

    __doc__ = _libcudart.cudaGetErrorString(55)


class cudaErrorProfilerNotInitialized(cudaError):
    """cudaErrorProfilerNotInitialized"""

    __doc__ = _libcudart.cudaGetErrorString(56)


class cudaErrorProfilerAlreadyStarted(cudaError):
    """cudaErrorProfilerAlreadyStarted"""

    __doc__ = _libcudart.cudaGetErrorString(57)


class cudaErrorProfilerAlreadyStopped(cudaError):
    """cudaErrorProfilerAlreadyStopped"""

    __doc__ = _libcudart.cudaGetErrorString(58)


class cudaErrorAssert(cudaError):
    """
    cudaErrorAssert

    An assert triggered in device code during kernel execution. The device
    cannot be used again. All existing allocations are invalid. To continue
    using CUDA, the process must be terminated and relaunched.
    """

    __doc__ = _libcudart.cudaGetErrorString(59)


class cudaErrorTooManyPeers(cudaError):
    """
    cudaErrorTooManyPeers

    This error indicates that the hardware resources required to enable
    peer access have been exhausted for one or more of the devices
    passed to cudaEnablePeerAccess().
    """

    __doc__ = _libcudart.cudaGetErrorString(60)


class cudaErrorHostMemoryAlreadyRegistered(cudaError):
    """
    cudaErrorHostMemoryAlreadyRegistered

    This error indicates that the memory range passed to cudaHostRegister()
    has already been registered.
    """

    __doc__ = _libcudart.cudaGetErrorString(61)


class cudaErrorHostMemoryNotRegistered(cudaError):
    """
    cudaErrorHostMemoryNotRegistered

    This error indicates that the pointer passed to cudaHostUnregister()
    does not correspond to any currently registered memory region.
    """

    __doc__ = _libcudart.cudaGetErrorString(62)


class cudaErrorOperatingSystem(cudaError):
    """
    cudaErrorOperatingSystem

    This error indicates that an OS call failed.
    """

    __doc__ = _libcudart.cudaGetErrorString(63)


class cudaErrorPeerAccessUnsupported(cudaError):
    """
    cudaErrorPeerAccessUnsupported

    This error indicates that P2P access is not supported across the
    given devices.
    """

    __doc__ = _libcudart.cudaGetErrorString(64)


class cudaErrorLaunchMaxDepthExceeded(cudaError):
    """
    cudaErrorLaunchMaxDepthExceeded

    This error indicates that a device runtime grid launch did not occur
    because the depth of the child grid would exceed the maximum supported
    number of nested grid launches.
    """

    __doc__ = _libcudart.cudaGetErrorString(65)


class cudaErrorLaunchFileScopedTex(cudaError):
    """
    cudaErrorLaunchFileScopedTex

    This error indicates that a grid launch did not occur because the
    kernel uses file-scoped textures which are unsupported by the device
    runtime. Kernels launched via the device runtime only support textures
    created with the Texture Object API's.
    """

    __doc__ = _libcudart.cudaGetErrorString(66)


class cudaErrorLaunchFileScopedSurf(cudaError):
    """
    cudaErrorLaunchFileScopedSurf

    This error indicates that a grid launch did not occur because the
    kernel uses file-scoped surfaces which are unsupported by the device
    runtime. Kernels launched via the device runtime only support surfaces
    created with the Surface Object API's.
    """

    __doc__ = _libcudart.cudaGetErrorString(67)


class cudaErrorSyncDepthExceeded(cudaError):
    """
    cudaErrorSyncDepthExceeded

    This error indicates that a call to cudaDeviceSynchronize made from
    the device runtime failed because the call was made at grid depth
    greater than than either the default (2 levels of grids) or user
    specified device limit cudaLimitDevRuntimeSyncDepth. To be able to
    synchronize on launched grids at a greater depth successfully, the
    maximum nested depth at which cudaDeviceSynchronize will be called
    must be specified with the cudaLimitDevRuntimeSyncDepth limit to the
    cudaDeviceSetLimit api before the host-side launch of a kernel using
    the device runtime. Keep in mind that additional levels of sync depth
    require the runtime to reserve large amounts of device memory that
    cannot be used for user allocations. Note that cudaDeviceSynchronize
    made from device runtime is only supported on devices of compute
    capability < 9.0.
    """

    __doc__ = _libcudart.cudaGetErrorString(68)


class cudaErrorLaunchPendingCountExceeded(cudaError):
    """
    cudaErrorLaunchPendingCountExceeded

    This error indicates that a device runtime grid launch failed because
    the launch would exceed the limit cudaLimitDevRuntimePendingLaunchCount.
    For this launch to proceed successfully, cudaDeviceSetLimit must be
    called to set the cudaLimitDevRuntimePendingLaunchCount to be higher
    than the upper bound of outstanding launches that can be issued to
    the device runtime. Keep in mind that raising the limit of pending
    device runtime launches will require the runtime to reserve device
    memory that cannot be used for user allocations.
    """

    __doc__ = _libcudart.cudaGetErrorString(69)


class cudaErrorNotPermitted(cudaError):
    """
    cudaErrorNotPermitted

    This error indicates the attempted operation is not permitted.
    """

    __doc__ = _libcudart.cudaGetErrorString(70)


class cudaErrorNotSupported(cudaError):
    """
    cudaErrorNotSupported

    This error indicates the attempted operation is not supported on the
    current system or device.
    """

    __doc__ = _libcudart.cudaGetErrorString(71)


class cudaErrorHardwareStackError(cudaError):
    """
    cudaErrorHardwareStackError

    Device encountered an error in the call stack during kernel execution,
    possibly due to stack corruption or exceeding the stack size limit.
    This leaves the process in an inconsistent state and any further CUDA
    work will return the same error. To continue using CUDA, the process
    must be terminated and relaunched.
    """

    __doc__ = _libcudart.cudaGetErrorString(72)


class cudaErrorIllegalInstruction(cudaError):
    """
    cudaErrorIllegalInstruction

    The device encountered an illegal instruction during kernel execution
    This leaves the process in an inconsistent state and any further CUDA
    work will return the same error. To continue using CUDA, the process
    must be terminated and relaunched.
    """

    __doc__ = _libcudart.cudaGetErrorString(73)


class cudaErrorMisalignedAddress(cudaError):
    """
    cudaErrorMisalignedAddress

    The device encountered a load or store instruction on a memory address
    which is not aligned. This leaves the process in an inconsistent state
    and any further CUDA work will return the same error. To continue using
    CUDA, the process must be terminated and relaunched.
    """

    __doc__ = _libcudart.cudaGetErrorString(74)


class cudaErrorInvalidAddressSpace(cudaError):
    """
    cudaErrorInvalidAddressSpace

    While executing a kernel, the device encountered an instruction which
    can only operate on memory locations in certain address spaces (global,
    shared, or local), but was supplied a memory address not belonging to
    an allowed address space. This leaves the process in an inconsistent
    state and any further CUDA work will return the same error. To continue
    using CUDA, the process must be terminated and relaunched.
    """

    __doc__ = _libcudart.cudaGetErrorString(75)


class cudaErrorInvalidPc(cudaError):
    """
    cudaErrorInvalidPc

    The device encountered an invalid program counter. This leaves the process
    in an inconsistent state and any further CUDA work will return the same
    error. To continue using CUDA, the process must be terminated and relaunched.
    """

    __doc__ = _libcudart.cudaGetErrorString(76)


class cudaErrorIllegalAddress(cudaError):
    """
    cudaErrorIllegalAddress

    The device encountered a load or store instruction on an invalid memory
    address. This leaves the process in an inconsistent state and any further
    CUDA work will return the same error. To continue using CUDA, the process
    must be terminated and relaunched.
    """

    __doc__ = _libcudart.cudaGetErrorString(77)


class cudaErrorInvalidPtx(cudaError):
    """
    cudaErrorInvalidPtx

    A PTX compilation failed. The runtime may fall back to compiling PTX
    if an application does not contain a suitable binary for the current
    device.
    """

    __doc__ = _libcudart.cudaGetErrorString(78)


class cudaErrorInvalidGraphicsContext(cudaError):
    """
    cudaErrorInvalidGraphicsContext

    This indicates an error with the OpenGL or DirectX context.
    """

    __doc__ = _libcudart.cudaGetErrorString(79)


class cudaErrorStartupFailure(cudaError):
    """
    cudaErrorStartupFailure

    This indicates an internal startup failure in the CUDA runtime.
    """

    __doc__ = _libcudart.cudaGetErrorString(127)


cudaExceptions = {
    1: cudaErrorMissingConfiguration,
    2: cudaErrorMemoryAllocation,
    3: cudaErrorInitializationError,
    4: cudaErrorLaunchFailure,
    5: cudaErrorPriorLaunchFailure,
    6: cudaErrorLaunchTimeout,
    7: cudaErrorLaunchOutOfResources,
    8: cudaErrorInvalidDeviceFunction,
    9: cudaErrorInvalidConfiguration,
    10: cudaErrorInvalidDevice,
    11: cudaErrorInvalidValue,
    12: cudaErrorInvalidPitchValue,
    13: cudaErrorInvalidSymbol,
    14: cudaErrorMapBufferObjectFailed,
    15: cudaErrorUnmapBufferObjectFailed,
    16: cudaErrorInvalidHostPointer,
    17: cudaErrorInvalidDevicePointer,
    18: cudaErrorInvalidTexture,
    19: cudaErrorInvalidTextureBinding,
    20: cudaErrorInvalidChannelDescriptor,
    21: cudaErrorInvalidMemcpyDirection,
    22: cudaError,
    23: cudaErrorTextureFetchFailed,
    24: cudaErrorTextureNotBound,
    25: cudaErrorSynchronizationError,
    26: cudaErrorInvalidFilterSetting,
    27: cudaErrorInvalidNormSetting,
    28: cudaErrorMixedDeviceExecution,
    29: cudaErrorCudartUnloading,
    30: cudaErrorUnknown,
    31: cudaErrorNotYetImplemented,
    32: cudaErrorMemoryValueTooLarge,
    33: cudaErrorInvalidResourceHandle,
    34: cudaErrorNotReady,
    35: cudaErrorInsufficientDriver,
    36: cudaErrorSetOnActiveProcess,
    37: cudaErrorInvalidSurface,
    38: cudaErrorNoDevice,
    39: cudaErrorECCUncorrectable,
    40: cudaErrorSharedObjectSymbolNotFound,
    41: cudaErrorSharedObjectInitFailed,
    42: cudaErrorUnsupportedLimit,
    43: cudaErrorDuplicateVariableName,
    44: cudaErrorDuplicateTextureName,
    45: cudaErrorDuplicateSurfaceName,
    46: cudaErrorDevicesUnavailable,
    47: cudaErrorInvalidKernelImage,
    48: cudaErrorNoKernelImageForDevice,
    49: cudaErrorIncompatibleDriverContext,
    50: cudaErrorPeerAccessAlreadyEnabled,
    51: cudaErrorPeerAccessNotEnabled,
    52: cudaError,
    53: cudaError,
    54: cudaErrorDeviceAlreadyInUse,
    55: cudaErrorProfilerDisabled,
    56: cudaErrorProfilerNotInitialized,
    57: cudaErrorProfilerAlreadyStarted,
    58: cudaErrorProfilerAlreadyStopped,
    59: cudaErrorAssert,
    60: cudaErrorTooManyPeers,
    61: cudaErrorHostMemoryAlreadyRegistered,
    62: cudaErrorHostMemoryNotRegistered,
    63: cudaErrorOperatingSystem,
    64: cudaErrorPeerAccessUnsupported,
    65: cudaErrorLaunchMaxDepthExceeded,
    66: cudaErrorLaunchFileScopedTex,
    67: cudaErrorLaunchFileScopedSurf,
    68: cudaErrorSyncDepthExceeded,
    69: cudaErrorLaunchPendingCountExceeded,
    70: cudaErrorNotPermitted,
    71: cudaErrorNotSupported,
    72: cudaErrorHardwareStackError,
    73: cudaErrorIllegalInstruction,
    74: cudaErrorMisalignedAddress,
    75: cudaErrorInvalidAddressSpace,
    76: cudaErrorInvalidPc,
    77: cudaErrorIllegalAddress,
    78: cudaErrorInvalidPtx,
    79: cudaErrorInvalidGraphicsContext,
    127: cudaErrorStartupFailure,
}


def cudaCheckStatus(status: int) -> None:
    """
    Raise CUDA exception.

    Raise an exception corresponding to the specified CUDA runtime error
    code.

    Parameters
    ----------
    status : int
        CUDA runtime error code.

    See Also
    --------
    cudaExceptions
    """

    if status != 0:
        try:
            e = cudaExceptions[status]
        except KeyError:
            raise cudaError("unknown CUDA error %s" % status)
        else:
            raise e


# Memory allocation functions (adapted from pystream):
_libcudart.cudaMalloc.restype = int
_libcudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]


def cudaMalloc(count: int, ctype: type | None = None) -> ctypes.c_void_p:
    """
    Allocate device memory.

    Allocate memory on the device associated with the current active
    context.

    Parameters
    ----------
    count : int
        Number of bytes of memory to allocate
    ctype : _ctypes.SimpleType, optional
        ctypes type to cast returned pointer.

    Returns
    -------
    ptr : ctypes pointer
        Pointer to allocated device memory.

    """

    ptr = ctypes.c_void_p()
    assert _libcudart
    status = _libcudart.cudaMalloc(ctypes.byref(ptr), count)
    cudaCheckStatus(status)
    if ctype is not None:
        ptr = ctypes.cast(ptr, ctypes.POINTER(ctype))
    return ptr  # pyright: ignore[reportReturnType]


_libcudart.cudaFree.restype = int
_libcudart.cudaFree.argtypes = [ctypes.c_void_p]


def cudaFree(ptr: ctypes.c_void_p) -> None:
    """
    Free device memory.

    Free allocated memory on the device associated with the current active
    context.

    Parameters
    ----------
    ptr : ctypes pointer
        Pointer to allocated device memory.

    """

    assert _libcudart
    status = _libcudart.cudaFree(ptr)
    cudaCheckStatus(status)


_libcudart.cudaMallocPitch.restype = int
_libcudart.cudaMallocPitch.argtypes = [
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.c_size_t,
    ctypes.c_size_t,
]


def cudaMallocPitch(pitch: int, rows: int, cols: int, elesize: int) -> tuple[ctypes.c_void_p, int]:
    """
    Allocate pitched device memory.

    Allocate pitched memory on the device associated with the current active
    context.

    Parameters
    ----------
    pitch : int
        Pitch for allocation.
    rows : int
        Requested pitched allocation height.
    cols : int
        Requested pitched allocation width.
    elesize : int
        Size of memory element.

    Returns
    -------
    ptr : ctypes pointer
        Pointer to allocated device memory.
    pitch : int
        The pitch of the allocated memory.

    """

    ptr = ctypes.c_void_p()
    assert _libcudart
    status = _libcudart.cudaMallocPitch(
        ctypes.byref(ptr), ctypes.c_size_t(pitch), cols * elesize, rows
    )
    cudaCheckStatus(status)
    return ptr, pitch


# Memory copy modes:
cudaMemcpyHostToHost = 0
cudaMemcpyHostToDevice = 1
cudaMemcpyDeviceToHost = 2
cudaMemcpyDeviceToDevice = 3
cudaMemcpyDefault = 4

_libcudart.cudaMemcpy.restype = int
_libcudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]


def cudaMemcpy_htod(dst: ctypes.c_void_p, src: ctypes.c_void_p, count: int) -> None:
    """
    Copy memory from host to device.

    Copy data from host memory to device memory.

    Parameters
    ----------
    dst : ctypes pointer
        Device memory pointer.
    src : ctypes pointer
        Host memory pointer.
    count : int
        Number of bytes to copy.

    """

    assert _libcudart
    status = _libcudart.cudaMemcpy(dst, src, ctypes.c_size_t(count), cudaMemcpyHostToDevice)
    cudaCheckStatus(status)


def cudaMemcpy_dtoh(dst: ctypes.c_void_p, src: ctypes.c_void_p, count: int) -> None:
    """
    Copy memory from device to host.

    Copy data from device memory to host memory.

    Parameters
    ----------
    dst : ctypes pointer
        Host memory pointer.
    src : ctypes pointer
        Device memory pointer.
    count : int
        Number of bytes to copy.

    """

    assert _libcudart
    status = _libcudart.cudaMemcpy(dst, src, ctypes.c_size_t(count), cudaMemcpyDeviceToHost)
    cudaCheckStatus(status)


_libcudart.cudaMemGetInfo.restype = int
_libcudart.cudaMemGetInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p]


def cudaMemGetInfo() -> tuple[int, int]:
    """
    Return the amount of free and total device memory.

    Returns
    -------
    free : long
        Free memory in bytes.
    total : long
        Total memory in bytes.

    """

    free = ctypes.c_size_t()
    total = ctypes.c_size_t()
    assert _libcudart
    status = _libcudart.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total))
    cudaCheckStatus(status)
    return free.value, total.value


_libcudart.cudaSetDevice.restype = int
_libcudart.cudaSetDevice.argtypes = [ctypes.c_int]


def cudaSetDevice(dev: int) -> None:
    """
    Set current CUDA device.

    Select a device to use for subsequent CUDA operations.

    Parameters
    ----------
    dev : int
        Device number.

    """

    assert _libcudart
    status = _libcudart.cudaSetDevice(dev)
    cudaCheckStatus(status)


_libcudart.cudaGetDevice.restype = int
_libcudart.cudaGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]


def cudaGetDevice() -> int:
    """
    Get current CUDA device.

    Return the identifying number of the device currently used to
    process CUDA operations.

    Returns
    -------
    dev : int
        Device number.

    """

    dev = ctypes.c_int()
    assert _libcudart
    status = _libcudart.cudaGetDevice(ctypes.byref(dev))
    cudaCheckStatus(status)
    return dev.value


_libcudart.cudaDriverGetVersion.restype = int
_libcudart.cudaDriverGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]


def cudaDriverGetVersion() -> int:
    """
    Get installed CUDA driver version.

    Return the version of the installed CUDA driver as an integer. If
    no driver is detected, 0 is returned.

    Returns
    -------
    version : int
        Driver version.
    """

    version = ctypes.c_int()
    assert _libcudart
    status = _libcudart.cudaDriverGetVersion(ctypes.byref(version))
    cudaCheckStatus(status)
    return version.value


_libcudart.cudaRuntimeGetVersion.restype = int
_libcudart.cudaRuntimeGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]


def cudaRuntimeGetVersion() -> int:
    """
    Get installed CUDA runtime version.

    Return the version of the installed CUDA runtime as an integer. If
    no driver is detected, 0 is returned.

    Returns
    -------
    version : int
        Runtime version.
    """

    version = ctypes.c_int()
    assert _libcudart
    status = _libcudart.cudaRuntimeGetVersion(ctypes.byref(version))
    cudaCheckStatus(status)
    return version.value


try:
    _cudart_version = cudaRuntimeGetVersion()
except BaseException:
    _cudart_version = 99999


class _cudart_version_req(object):
    """
    Required CUDA Runtime decorator

    Decorator to replace function with a placeholder that raises an exception
    if the installed CUDA Runtime version is not greater than `v`.
    """

    def __init__(self, v: int | float) -> None:
        self.vs = str(v)
        if isinstance(v, int):
            major = str(v)
            minor = "0"
        else:
            match = re.search(r"(\d+)\.(\d+)", self.vs)
            assert match
            major, minor = match.groups()
        self.vi = int(major.ljust(len(major) + 1, "0") + minor.ljust(2, "0"))

    def __call__[T: Callable](self, f: T) -> T:
        @functools.wraps(f)
        def f_new(*args: Any, **kwargs: Any) -> Any:
            raise NotImplementedError("CUDART " + self.vs + " required")

        f_new.__doc__ = f.__doc__

        if _cudart_version >= self.vi:
            return f
        else:
            return f_new  # pyright: ignore[reportReturnType]


# Memory types:
cudaMemoryTypeHost = 1
cudaMemoryTypeDevice = 2


class cudaPointerAttributes(ctypes.Structure):
    """CUDA pointer attributes structure."""

    _fields_ = [
        ("memoryType", ctypes.c_int),
        ("device", ctypes.c_int),
        ("devicePointer", ctypes.c_void_p),
        ("hostPointer", ctypes.c_void_p),
    ]


_libcudart.cudaPointerGetAttributes.restype = int
_libcudart.cudaPointerGetAttributes.argtypes = [ctypes.c_void_p, ctypes.c_void_p]


def cudaPointerGetAttributes(ptr: ctypes.c_void_p) -> tuple[int, int]:
    """
    Get memory pointer attributes.

    Returns attributes of the specified pointer.

    Parameters
    ----------
    ptr : ctypes pointer
        Memory pointer to examine.

    Returns
    -------
    memory_type : int
        Memory type; 1 indicates host memory, 2 indicates device
        memory.
    device : int
        Number of device associated with pointer.

    Notes
    -----
    This function only works with CUDA 4.0 and later.

    """

    attributes = cudaPointerAttributes()
    assert _libcudart
    status = _libcudart.cudaPointerGetAttributes(ctypes.byref(attributes), ptr)
    cudaCheckStatus(status)
    return attributes.memoryType, attributes.device
