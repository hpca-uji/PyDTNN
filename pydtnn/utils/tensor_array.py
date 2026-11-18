"""TensorArray module for managing GPU-resident tensors with cuDNN descriptors."""

from __future__ import annotations

import copy
import ctypes
import logging
import re
import typing
from enum import StrEnum, auto
from typing import Any, Literal

import numpy as np

from pydtnn.utils.gpu import cudnn_dtype_from_numpy
from pydtnn.utils.constants import ArrayShape
from pydtnn.utils.tensor import TensorFormat, decode_shape, encode_shape

__all__ = ("TensorArray",)

logger = logging.getLogger(__name__)


if typing.TYPE_CHECKING:
    from pycuda.driver import Stream

type Ary = Any


class FormatType(StrEnum):
    """Enumeration of supported format types."""

    TENSOR = auto()
    FILTER = auto()
    SEQ = auto()
    NONE = auto()


class MemoryType(StrEnum):
    """Enumeration of supported memory types."""

    HOST = auto()
    GLOBAL = auto()
    DEVICE = auto()


class TensorArray[S: tuple, D: np.dtype]:  # noqa: D101
    """Wrapper class for GPU arrays that integrates with cuDNN descriptors."""

    @classmethod
    def wrap(
        cls,
        object: Any,
        *,
        tensor_format: TensorFormat = TensorFormat.NCHW,
    ) -> TensorArray[tuple[()], np.dtype[np.object_]]:
        """Wraps a object as a TensorArray"""
        return cls(
            np.array(object, dtype=np.object_, ndmax=1),
            host_shape=True,
            tensor_format=tensor_format,
            mem_type=MemoryType.HOST,
            fmt_type=FormatType.NONE
        )  # pyright: ignore[reportReturnType]

    @classmethod
    def zeros(
        cls,
        shape: S,
        dtype: D,
        *,
        host_shape: bool = False,
        tensor_format: TensorFormat = TensorFormat.NCHW,
        fmt_type: FormatType = FormatType.NONE,
        mem_type: MemoryType = MemoryType.DEVICE
    ) -> TensorArray[S, D]:
        """Creates a zero-initialized TensorArray."""
        return cls.asarray(
            np.zeros(shape, dtype),
            host_shape=True if host_shape else False,
            tensor_format=tensor_format,
            fmt_type=fmt_type,
            mem_type=mem_type
        )

    @classmethod
    def asarray(
        cls,
        /,
        ary: np.ndarray[S, D],
        *,
        host_shape: bool = False,
        tensor_format: TensorFormat = TensorFormat.NCHW,
        fmt_type: FormatType = FormatType.NONE,
        mem_type: MemoryType = MemoryType.DEVICE
    ) -> TensorArray[S, D]:
        """Creates a zero-initialized TensorArray."""
        match mem_type:
            case MemoryType.GLOBAL:
                from pycuda import driver as pycuda_driver
                gpu_arr = pycuda_driver.aligned_zeros(ary.shape, ary.dtype)
                gpu_arr = pycuda_driver.register_host_memory(gpu_arr, flags=pycuda_driver.mem_host_register_flags.DEVICEMAP)
                np.copyto(gpu_arr, ary)
            case MemoryType.DEVICE:
                from pycuda import gpuarray
                gpu_arr = gpuarray.to_gpu(ary)
            case MemoryType.HOST:
                gpu_arr = ary
        return cls(
            gpu_arr,
            host_shape=True if host_shape else ary.shape,
            tensor_format=tensor_format,
            fmt_type=fmt_type,
            mem_type=mem_type
        )

    def __init__(
        self,
        ary: Ary,
        *,
        host_shape: S | bool = False,
        tensor_format: TensorFormat = TensorFormat.NCHW,
        mem_type: MemoryType = MemoryType.DEVICE,
        fmt_type: FormatType = FormatType.NONE,
        desc: int | None = None,
    ) -> None:
        """Initializes the TensorArray instance."""

        self.ary: Ary
        self.host_shape: S = ary.shape
        self.tensor_format = TensorFormat(tensor_format.lower())
        self.mem_type = mem_type
        self.fmt_type = fmt_type
        self.desc: int = desc  # pyright: ignore[reportAttributeAccessIssue]

        if host_shape is True:
            self.ary = ary
        else:
            self.ary = self._expand_ary(ary)

        if desc:
            self.desc = desc
        elif self.size > 0:
            self._desc_init()

    @property
    def cudnn_dtype(self) -> int:
        """Get cuDNN dtype"""
        return cudnn_dtype_from_numpy(self.ary.dtype)

    @property
    def cudnn_tensor_format(self) -> int:
        """Get cuDNN integer constant for the current tensor format"""
        from pydtnn.libs import cudnn
        return cudnn.cudnnTensorFormat[f"CUDNN_TENSOR_{self.tensor_format.upper()}"]

    def _encode_shape(self, shape: ArrayShape) -> ArrayShape:
        """Encodes shape based on tensor format."""
        return encode_shape(shape, self.tensor_format)

    def _decode_shape(self, shape: ArrayShape) -> ArrayShape:
        """Decodes shape based on tensor format."""
        return decode_shape(shape, self.tensor_format)

    def _expand_ary(self, ary: Ary) -> None:
        """Expand backing array and reshape to 4D for cuDNN compatibility."""
        match len(ary.shape):
            case 1:
                match self.tensor_format:
                    case TensorFormat.NCHW:
                        shape = (1, ary.shape[0], 1, 1)
                    case TensorFormat.NHWC:
                        shape = (1, 1, 1, ary.shape[0])
                    case tensor_format:
                        raise NotImplementedError(f"Unsupported tensor format {tensor_format}!")
            case 2:
                match self.tensor_format:
                    case TensorFormat.NCHW:
                        shape = (ary.shape[0], ary.shape[1], 1, 1)
                    case TensorFormat.NHWC:
                        shape = (ary.shape[0], 1, 1, ary.shape[1])
                    case tensor_format:
                        raise NotImplementedError(f"Unsupported tensor format {tensor_format}!")
            case 3:
                match self.tensor_format:
                    case TensorFormat.NCHW:
                        shape = (ary.shape[0], 1, ary.shape[1], ary.shape[2])
                    case TensorFormat.NHWC:
                        shape = (ary.shape[0], ary.shape[1], ary.shape[2], 1)
                    case tensor_format:
                        raise NotImplementedError(f"Unsupported tensor format {tensor_format}!")
            case 4:
                shape = ary.shape
            case _:
                raise ValueError(
                    "The expected len shape are 1, 2, 3 or 4. Shape received:"
                    f" {len(ary.shape)}."
                )

        return ary.reshape(shape)

    def __eq__(self, value: object) -> bool:
        """Checks equality with another TensorArray."""
        if not isinstance(value, TensorArray):
            return False
        return (
            value.fmt_type == self.fmt_type
            and value.mem_type == self.mem_type
            and value.tensor_format == self.tensor_format
            and value.ary == self.ary
        )

    def __repr__(self) -> str:
        """Returns string representation of the TensorArray."""
        data = re.sub(r"[\s\n]+", " ", repr(self.ary))
        return f"<{self.__class__.__name__} format={self.fmt_type} memory={self.mem_type} data={data}>"

    @property
    def ptr(self) -> ctypes.c_void_p:
        """Returns a integer as a pointer to the underlying memory"""
        match self.mem_type:
            case MemoryType.DEVICE:
                ptr = int(self.ary.gpudata)
            case MemoryType.GLOBAL:
                ptr = int(self.ary.base.get_device_pointer())
            case MemoryType.HOST:
                ptr = int(self.ary.ctypes.data)
            case mem_type:
                raise TypeError(f"Unsupported memory type {mem_type}")
        return ctypes.c_void_p(ptr)

    def _desc_init(self) -> None:
        """Initializes the cuDNN descriptor."""
        match self.fmt_type:
            case FormatType.TENSOR:
                from pydtnn.libs import cudnn
                n, c, h, w = self._decode_shape(self.shape)
                desc = cudnn.cudnnCreateTensorDescriptor()
                assert desc
                self.desc = desc
                cudnn.cudnnSetTensor4dDescriptor(
                    self.desc, self.cudnn_tensor_format, self.cudnn_dtype, n, c, h, w
                )
            case FormatType.FILTER:
                from pydtnn.libs import cudnn
                n, c, h, w = self._decode_shape(self.shape)
                desc = cudnn.cudnnCreateFilterDescriptor()
                assert desc
                self.desc = desc
                cudnn.cudnnSetFilter4dDescriptor(
                    self.desc, self.cudnn_dtype, self.cudnn_tensor_format, n, c, h, w
                )
            case FormatType.SEQ:
                from pydtnn.libs import cudnn
                desc = cudnn.cudnnCreateSeqDataDescriptor()
                assert desc
                self.desc = desc
                dim_a = [0, 0, 0, 0]
                dim_a[cudnn.cudnnSeqDataAxis["CUDNN_SEQDATA_BATCH_DIM"]] = self.shape[0]
                dim_a[cudnn.cudnnSeqDataAxis["CUDNN_SEQDATA_BEAM_DIM"]] = self.shape[1]
                dim_a[cudnn.cudnnSeqDataAxis["CUDNN_SEQDATA_TIME_DIM"]] = self.shape[2]
                dim_a[cudnn.cudnnSeqDataAxis["CUDNN_SEQDATA_VECT_DIM"]] = self.shape[3]
                axes = [0, 0, 0, 0]
                axes[0] = cudnn.cudnnSeqDataAxis["CUDNN_SEQDATA_BATCH_DIM"]
                axes[1] = cudnn.cudnnSeqDataAxis["CUDNN_SEQDATA_BEAM_DIM"]
                axes[2] = cudnn.cudnnSeqDataAxis["CUDNN_SEQDATA_TIME_DIM"]
                axes[3] = cudnn.cudnnSeqDataAxis["CUDNN_SEQDATA_VECT_DIM"]
                self.seq_length_array = (self.shape[-2],) * (self.shape[0] * self.shape[1])
                cudnn.cudnnSetSeqDataDescriptor(
                    self.desc,
                    self.cudnn_dtype,
                    4,
                    tuple(dim_a),
                    tuple(axes),
                    len(self.seq_length_array),
                    self.seq_length_array,
                    None,
                )
            case FormatType.NONE:
                self.desc = -1

            case tensor_type:
                raise NotImplementedError(f"Tensor type not implemented! ({tensor_type})")

    def _del_desc(self) -> None:
        """Destroys the cuDNN descriptor."""
        match self.fmt_type:
            case FormatType.TENSOR:
                from pydtnn.libs import cudnn
                cudnn.cudnnDestroyTensorDescriptor(self.desc)
            case FormatType.FILTER:
                from pydtnn.libs import cudnn
                cudnn.cudnnDestroyFilterDescriptor(self.desc)
            case FormatType.SEQ:
                from pydtnn.libs import cudnn
                cudnn.cudnnDestroySeqDataDescriptor(self.desc)
            case FormatType.NONE:
                pass
            case tensor_type:
                raise NotImplementedError(f"Tensor type not implemented! ({tensor_type})")
        self.desc = -1

    def __getattr__(self, name: str) -> Any:
        """Delegates attribute access to the underlying GPUArray."""
        return getattr(self.ary, name)

    def flatten(self, order: Literal["C", "F", "A"] = "C") -> TensorArray:
        """Returns a flattened view of the TensorArray."""
        return self._view(self.ary.reshape((-1,), order))  # pyright: ignore[reportReturnType]

    def reshape[NS: tuple = ArrayShape](
        self, shape: NS, order: Literal["C", "F", "A"] = "C"
    ) -> TensorArray[NS, D]:
        """Returns a reshaped view of the TensorArray."""
        return self._view(self.ary.reshape(shape, order))  # pyright: ignore[reportReturnType]

    def squeeze(self, dtype: np.dtype | None = None) -> TensorArray:
        """Squeeze operation is not supported for TensorArray."""
        raise ValueError("Can't squeeze TensorArray")

    def set(self, value: np.ndarray) -> None:
        """Copies data from CPU to GPU."""
        value = np.asarray(value.reshape(self.ary.shape), dtype=self.ary.dtype)
        match self.mem_type:
            case MemoryType.DEVICE:
                self.ary.set(value)
            case MemoryType.GLOBAL | MemoryType.HOST:
                self.ary[:] = value
            case mem_type:
                raise TypeError(f"Unsupported memory type {mem_type}")

    def set_async(self, value: np.ndarray, stream: Stream | None = None) -> None:
        """Asynchronously copies data from CPU to GPU."""
        value = np.asarray(value.reshape(self.ary.shape), dtype=self.ary.dtype)
        try:
            self.ary.set_async(value, stream)
        except AttributeError:
            self.set(value)  # sync fallback

    def get(self, ary: Ary | None = None) -> np.ndarray[S, D]:
        """Copies data from GPU to CPU and removes padding."""
        match self.mem_type:
            case MemoryType.DEVICE:
                value = self.ary.get()
            case MemoryType.GLOBAL | MemoryType.HOST:
                value = np.asarray(self.ary, copy=True)
            case mem_type:
                raise TypeError(f"Invalid memory type {mem_type}")

        if value.shape != self.host_shape:
            match len(self.host_shape):
                case 1:
                    match self.tensor_format:
                        case TensorFormat.NCHW:
                            value = np.squeeze(value, axis=(0, 2, 3))
                        case TensorFormat.NHWC:
                            value = np.squeeze(value, axis=(0, 1, 2))
                        case tensor_format:
                            raise NotImplementedError(f"Unsupported tensor format {tensor_format}!")
                case 2:
                    match self.tensor_format:
                        case TensorFormat.NCHW:
                            value = np.squeeze(value, axis=(2, 3))
                        case TensorFormat.NHWC:
                            value = np.squeeze(value, axis=(1, 2))
                        case tensor_format:
                            raise NotImplementedError(f"Unsupported tensor format {tensor_format}!")
                case 3:
                    match self.tensor_format:
                        case TensorFormat.NCHW:
                            value = np.squeeze(value, axis=(1,))
                        case TensorFormat.NHWC:
                            raise NotImplementedError(
                                "Shape padding not implemented for 3-dim shape on NHWC"
                            )
                        case tensor_format:
                            raise NotImplementedError(f"Unsupported tensor format {tensor_format}!")
                case 4:
                    value = value
                case dims:
                    raise ValueError(
                        f"The expected len shape are 1, 2, 3 or 4. Shape received: {dims}."
                    )

        if ary is None:
            return value
        else:
            ary[:] = value
            return None  # pyright: ignore[reportReturnType]

    def get_async(
        self,
        stream: Stream | None = None,
        ary: np.ndarray[ArrayShape, D] | None = None,
    ) -> None:
        """Asynchronously copies data from GPU to CPU."""
        if ary is None:
            raise ValueError("Destination must be defined!")
        if not hasattr(ary, "dtype"):
            raise TypeError("Destination must be a array!")
        if ary.dtype != self.ary.dtype:
            raise TypeError("Destination must be same dtype!")
        ary = ary.reshape(self.ary.shape)

        match self.mem_type:
            case MemoryType.DEVICE:
                self.ary.get_async(stream, ary)
            case MemoryType.GLOBAL | MemoryType.HOST:
                self.get(ary)  # sync fallback
            case mem_type:
                raise TypeError(f"Invalid memory type {mem_type}")

    def __array__(
        self, dtype: np.dtype | None = None, *, copy: bool | None = None
    ) -> np.ndarray[S, D]:
        """Converts TensorArray to a NumPy array."""
        if copy is False:
            raise ValueError("Must copy array")
        return np.asarray(self.get(), dtype=dtype)  # pyright: ignore[reportReturnType]

    def _view(self, ary: Ary, keep_shape: bool = True) -> TensorArray[S, D]:
        """Creates a new TensorArray instance sharing the same underlying configuration."""
        # NOTE: In some cases, it would be possible to share the descriptor more
        # aggressively, but we don't have enough information to decide when.
        # NOTE: Currently the descriptor is never shared, but if this is changed
        # the descriptor destruction handling must be changed to mirror it.
        return self.__class__(
            ary,
            host_shape=self.host_shape if keep_shape else False,
            tensor_format=self.tensor_format,
            mem_type=self.mem_type,
            fmt_type=self.fmt_type,
        )

    def copy(self) -> TensorArray[S, D]:
        """Returns a deep copy of the TensorArray."""
        return copy.deepcopy(self)

    def __copy__(self) -> TensorArray[S, D]:
        """Returns a shallow copy of the TensorArray."""
        return self._view(self.ary)

    def __deepcopy__(self, memo: dict) -> TensorArray[S, D]:
        """Returns a deep copy of the TensorArray."""
        obj = memo[id(self)] = self.__class__(
            copy.deepcopy(self.ary, memo),
            host_shape=self.host_shape,
            tensor_format=self.tensor_format,
            mem_type=self.mem_type,
            fmt_type=self.fmt_type,
        )
        return obj

    def close(self) -> None:
        """Releases resources associated with the TensorArray."""
        if self.ary is not None:
            self._del_desc()
            self.ary = None
        self.desc = -1

    def __del__(self) -> None:
        """Finalizer to ensure resources are released."""
        try:
            self.close()
        except:  # noqa: E722
            pass

    def __len__(self) -> int:
        """Returns the length of the underlying array."""
        return len(self.ary)

    def _operable(
        self, other: Ary
    ) -> Ary:
        """Prepares operand for arithmetic operations."""
        if isinstance(other, TensorArray):
            other = other.ary
        if hasattr(other, "dtype") and other.dtype != self.ary.dtype:
            other = other.astype(self.ary.dtype)
        return other

    def __add__(self, other: Ary) -> TensorArray[S, D]:
        """Addition operator."""
        other = self._operable(other)
        return self._view(self.ary.__add__(other))

    def __radd__(self, other: Ary) -> TensorArray[S, D]:
        """Right-side addition operator."""
        other = self._operable(other)
        return self._view(self.ary.__radd__(other))

    def __sub__(self, other: Ary) -> TensorArray[S, D]:
        """Subtraction operator."""
        other = self._operable(other)
        return self._view(self.ary.__sub__(other))

    def __rsub__(self, other: Ary) -> TensorArray[S, D]:
        """Right-side subtraction operator."""
        other = self._operable(other)
        return self._view(self.ary.__rsub__(other))

    def __iadd__(self, other: Ary) -> TensorArray[S, D]:
        """In-place addition operator."""
        other = self._operable(other)
        return self._view(self.ary.__iadd__(other))

    def __isub__(self, other: Ary) -> TensorArray[S, D]:
        """In-place subtraction operator."""
        other = self._operable(other)
        return self._view(self.ary.__isub__(other))

    def __neg__(self, other: Ary) -> TensorArray[S, D]:
        """Negation operator."""
        other = self._operable(other)
        return self._view(self.ary.__neg__(other))

    def __mul__(self, other: Ary) -> TensorArray[S, D]:
        """Multiplication operator."""
        other = self._operable(other)
        return self._view(self.ary.__mul__(other))

    def __rmul__(self, other: Ary) -> TensorArray[S, D]:
        """Right-side multiplication operator."""
        other = self._operable(other)
        return self._view(self.ary.__rmul__(other))

    def __truediv__(self, other: Ary) -> TensorArray[S, D]:
        """Division operator."""
        other = self._operable(other)
        return self._view(self.ary.__truediv__(other))

    def __rtruediv__(
        self, other: Ary
    ) -> TensorArray[S, D]:
        """Right-side division operator."""
        other = self._operable(other)
        return self._view(self.ary.__rtruediv__(other))

    def __pow__(self, other: Ary) -> TensorArray[S, D]:
        """Power operator."""
        other = self._operable(other)
        return self._view(self.ary.__pow__(other))

    def __rpow__(self, other: Ary) -> TensorArray[S, D]:
        """Right-side power operator."""
        other = self._operable(other)
        return self._view(self.ary.__rpow__(other))

    def __getitem__(self, index: int | tuple | slice | np.ndarray) -> TensorArray[S, D]:
        """Indexing operator."""
        return self._view(self.ary.__getitem__(index), keep_shape=False)

    def __setitem__(
        self,
        key: int | tuple | slice | np.ndarray,
        value: Ary,
    ) -> None:
        """Set item operator."""
        value = self._operable(value)
        return self.ary.__setitem__(key, value)

    def __abs__(self) -> TensorArray[S, D]:
        """Absolute value operator."""
        return self._view(self.ary.__abs__())
