from pydtnn.utils.tensor import TensorFormat, encode_shape, decode_shape
from pydtnn.utils.constants import ArrayShape
import numpy as np
from enum import StrEnum, auto
import copy
import ctypes
import logging
logger = logging.getLogger(__name__)


try:
    from pycuda import gpuarray  # type: ignore
    from pycuda import driver as pycuda_driver  # type: ignore
    from pydtnn.libs import cudnn as cudnn
except Exception:
    pass


class TensorArray:

    class TensorType(StrEnum):
        TENSOR = auto()
        FILTER = auto()
        SEQ = auto()
        OTHER = auto()
    # ---  END EnumTensorType --- #

    @staticmethod
    def new_empty(shape: ArrayShape, dtype: np.dtype,
                  tensor_format: TensorFormat, cudnn_dtype: int,
                  tensor_type: TensorType = TensorType.TENSOR, desc: int | None = None,
                  gpudirect: bool = False, cublas: bool = False):
        gpu_arr = gpuarray.empty(shape, dtype)
        return TensorArray(gpu_arr=gpu_arr, tensor_format=tensor_format, cudnn_dtype=cudnn_dtype,
                           tensor_type=tensor_type, desc=desc, gpudirect=gpudirect, cublas=cublas)

    @staticmethod
    def new_zeros(shape: ArrayShape, dtype: np.dtype,
                  tensor_format: TensorFormat, cudnn_dtype: int,
                  tensor_type: TensorType = TensorType.TENSOR, desc: int | None = None,
                  gpudirect: bool = False, cublas: bool = False):
        gpu_arr = gpuarray.zeros(shape, dtype)
        return TensorArray(gpu_arr=gpu_arr, tensor_format=tensor_format, cudnn_dtype=cudnn_dtype,
                           tensor_type=tensor_type, desc=desc, gpudirect=gpudirect, cublas=cublas)

    @staticmethod
    def new_pair_gpudirect(drv: "pycuda_driver", shape: ArrayShape, dtype: np.dtype,
                           tensor_format: TensorFormat, cudnn_dtype: int,
                           tensor_type: TensorType = TensorType.TENSOR,
                           desc: int | None = None, gpudirect: bool = False, cublas: bool = False) -> tuple[np.ndarray, "TensorArray"]:
        x_cpu = drv.aligned_zeros(shape, dtype)
        x_gpu = drv.register_host_memory(x_cpu, flags=drv.mem_host_register_flags.DEVICEMAP)

        x_gpu = TensorArray(x_gpu, tensor_format=tensor_format, cudnn_dtype=cudnn_dtype, tensor_type=tensor_type,
                            desc=desc, gpudirect=gpudirect, cublas=cublas)
        return (x_cpu, x_gpu)

    @staticmethod
    def new_pair(shape: ArrayShape, dtype: np.dtype,
                 tensor_format: TensorFormat, cudnn_dtype: int,
                 tensor_type: TensorType = TensorType.TENSOR,
                 desc: int | None = None, gpudirect: bool = False, cublas: bool = False) -> tuple[np.ndarray, "TensorArray"]:
        x_cpu = np.zeros(shape, dtype)
        x_gpu = gpuarray.zeros(shape, dtype)
        x_gpu = TensorArray(x_gpu, tensor_format=tensor_format, cudnn_dtype=cudnn_dtype, tensor_type=tensor_type,
                            desc=desc, gpudirect=gpudirect, cublas=cublas)
        return (x_cpu, x_gpu)

    @staticmethod
    def new(shape: ArrayShape, dtype: np.dtype,
            tensor_format: TensorFormat, cudnn_dtype: int,
            tensor_type: TensorType = TensorType.TENSOR,
            desc: int | None = None, gpudirect: bool = False, cublas: bool = False,
            drv: "pycuda_driver" = None) -> tuple[np.ndarray, "TensorArray"]:
        if drv is not None:
            return TensorArray.new_pair_gpudirect(drv=drv, shape=shape,
                                                  dtype=dtype, tensor_format=tensor_format,
                                                  cudnn_dtype=cudnn_dtype, tensor_type=tensor_type,
                                                  desc=desc, gpudirect=gpudirect, cublas=cublas)
        else:
            return TensorArray.new_pair(shape=shape, dtype=dtype, tensor_format=tensor_format,
                                        cudnn_dtype=cudnn_dtype, tensor_type=tensor_type,
                                        desc=desc, gpudirect=gpudirect, cublas=cublas)

    # ---

    def __init__(self, gpu_arr: "gpuarray.GPUArray", tensor_format: TensorFormat, cudnn_dtype: int,
                 tensor_type: TensorType = TensorType.TENSOR, desc: int | None = None,
                 gpudirect: bool = False, cublas: bool = False):

        self.tensor_format = TensorFormat(tensor_format.lower())
        self.cudnn_dtype = cudnn_dtype
        self.tensor_type = tensor_type
        self.gpudirect = gpudirect
        self.cublas = cublas

        # ---
        self.ary: gpuarray.GPUArray
        self.desc: int = -1

        self._set_ary(gpu_arr)

        if desc:
            self.desc = desc
        elif self.size > 0:
            self._desc_init()

    @property
    def cudnn_tensor_format(self) -> int:
        return cudnn.cudnnTensorFormat['CUDNN_TENSOR_' + self.tensor_format.upper()]

    def _encode_shape(self, shape):
        return encode_shape(shape, self.tensor_format)

    def _decode_shape(self, shape):
        return decode_shape(shape, self.tensor_format)

    def _set_ary(self, gpu_arr: "gpuarray.GPUArray") -> None:
        """Set backing gpu array"""
        match len(gpu_arr.shape):
            case 1:
                match self.tensor_format:
                    case TensorFormat.NCHW:
                        shape = (1, *gpu_arr.shape, 1, 1)
                    case TensorFormat.NHWC:
                        shape = (1, 1, 1, *gpu_arr.shape)
                    case tensor_format:
                        raise NotImplementedError(f"Unsupported tensor format {tensor_format}!")
            case 2:
                match self.tensor_format:
                    case TensorFormat.NCHW:
                        shape = (*gpu_arr.shape, 1, 1)
                    case TensorFormat.NHWC:
                        shape = (gpu_arr.shape[0], 1, 1, gpu_arr.shape[1])
                    case tensor_format:
                        raise NotImplementedError(f"Unsupported tensor format {tensor_format}!")
            case 3:
                match self.tensor_format:
                    case TensorFormat.NCHW:
                        shape = (gpu_arr.shape[0], 1, gpu_arr.shape[1], gpu_arr.shape[2])
                    case TensorFormat.NHWC:
                        raise NotImplementedError("Shape padding not implemented for 3-dim shape on NHWC")
            case 4:
                shape = gpu_arr.shape
            case _:
                raise ValueError(f"The expected len shape are 1, 2, 3 or 4. Shape received: {len(gpu_arr.shape)}.")

        self.shape_cpu = gpu_arr.shape
        self.ary = gpu_arr.reshape(shape)
    # ---

    @property
    def ptr_voidp(self) -> ctypes.c_void_p:
        if self.gpudirect:
            return ctypes.c_void_p(int(self.ary.base.get_device_pointer()))
        else:
            return ctypes.c_void_p(int(self.ary.gpudata))

    @property
    def ptr_intp(self) -> np.intp:
        return np.intp(self.ary.base.get_device_pointer())

    def _desc_init(self) -> None:
        match self.tensor_type:
            case self.TensorType.TENSOR:
                n, c, h, w = self._decode_shape(self.shape)
                desc = cudnn.cudnnCreateTensorDescriptor()
                assert desc
                self.desc = desc
                cudnn.cudnnSetTensor4dDescriptor(self.desc, self.cudnn_tensor_format,
                                                 self.cudnn_dtype, n, c, h, w)
            case self.TensorType.FILTER:
                n, c, h, w = self._decode_shape(self.shape)
                desc = cudnn.cudnnCreateFilterDescriptor()
                assert desc
                self.desc = desc
                cudnn.cudnnSetFilter4dDescriptor(self.desc, self.cudnn_dtype,
                                                 self.cudnn_tensor_format, n, c, h, w)
            case self.TensorType.SEQ:
                desc = cudnn.cudnnCreateSeqDataDescriptor()
                assert desc
                self.desc = desc
                dimA = np.array([0, 0, 0, 0], dtype=np.int32)
                dimA[cudnn.cudnnSeqDataAxis["CUDNN_SEQDATA_BATCH_DIM"]] = self.shape[0]
                dimA[cudnn.cudnnSeqDataAxis["CUDNN_SEQDATA_BEAM_DIM"]] = self.shape[1]
                dimA[cudnn.cudnnSeqDataAxis["CUDNN_SEQDATA_TIME_DIM"]] = self.shape[2]
                dimA[cudnn.cudnnSeqDataAxis["CUDNN_SEQDATA_VECT_DIM"]] = self.shape[3]
                axes = np.array([0, 0, 0, 0], dtype=np.int32)
                axes[0] = cudnn.cudnnSeqDataAxis["CUDNN_SEQDATA_BATCH_DIM"]
                axes[1] = cudnn.cudnnSeqDataAxis["CUDNN_SEQDATA_BEAM_DIM"]
                axes[2] = cudnn.cudnnSeqDataAxis["CUDNN_SEQDATA_TIME_DIM"]
                axes[3] = cudnn.cudnnSeqDataAxis["CUDNN_SEQDATA_VECT_DIM"]
                self.seq_length_array = np.full(shape=(self.shape[0] * self.shape[1]), fill_value=self.shape[-2], dtype=np.int32)
                # print(self.shape, dimA, axes, len(seq_length_array))
                cudnn.cudnnSetSeqDataDescriptor(self.desc, self.cudnn_dtype,
                                                np.int32(4), dimA, axes,
                                                np.int32(len(self.seq_length_array)), self.seq_length_array,
                                                None)
            case self.TensorType.OTHER:
                pass  # do nothing.

            case tensor_type:
                raise NotImplementedError(f"Tensor type not implemented! ({tensor_type})")
    # ---

    def _del_desc(self) -> None:
        match self.tensor_type:
            case self.TensorType.TENSOR:
                cudnn.cudnnDestroyTensorDescriptor(self.desc)
            case self.TensorType.FILTER:
                cudnn.cudnnDestroyFilterDescriptor(self.desc)
            case self.TensorType.SEQ:
                cudnn.cudnnDestroySeqDataDescriptor(self.desc)
            case self.TensorType.OTHER:
                pass  # do nothing.
            case tensor_type:
                raise NotImplementedError(f"Tensor type not implemented! ({tensor_type})")
        self.desc = -1
    # ---

    def __getattr__(self, name):
        return getattr(self.ary, name)
    # ---

    def reshape(self, shape, order="C") -> "TensorArray":
        """Reshape TensorArray"""
        return self._view(self.ary.reshape(shape, order))

    def squeeze(self, dtype=None) -> "TensorArray":
        """Squeeze TensorArray"""
        raise ValueError("Can't squeeze TensorArray")

    def set(self, value: np.ndarray) -> None:
        """CPU to GPU with expand_dims"""
        self.ary.set(value.reshape(self.ary.shape))

    def get(self, ary=None) -> np.ndarray:
        """GPU to CPU with squeeze"""
        value = self.ary.get()

        match len(self.cpu_shape):
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
                        raise NotImplementedError("Shape padding not implemented for 3-dim shape on NHWC")
            case 4:
                value = value
            case _:
                raise ValueError(f"The expected len shape are 1, 2, 3 or 4. Shape received: {len(self.ary.shape)}.")

        if ary is None:
            return value
        else:
            ary[:] = value
            return None  # type: ignore

    def __array__(self, dtype=None, *, copy=None):
        """ NumPy cast helper """
        return np.asarray(self.get(), dtype=dtype)

    def _view(self, ary):
        """TensorArray view"""
        return TensorArray(gpu_arr=ary,
                           tensor_format=self.tensor_format,
                           cudnn_dtype=self.cudnn_dtype,
                           tensor_type=self.tensor_type,
                           gpudirect=self.gpudirect,
                           cublas=self.cublas,
                           desc=self.desc)

    def copy(self):
        """ NumPy-like copy. """
        return copy.deepcopy(self)

    def __copy__(self):
        """Shallow copy"""
        return self._view(self.ary)

    def __deepcopy__(self, memo: dict):
        """Deep copy"""
        obj = TensorArray(gpu_arr=copy.deepcopy(self.ary, memo),
                          tensor_format=self.tensor_format,
                          cudnn_dtype=self.cudnn_dtype,
                          tensor_type=self.tensor_type,
                          gpudirect=self.gpudirect,
                          cublas=self.cublas,
                          desc=-1)
        memo[id(self)] = obj
        return obj

    def close(self) -> None:
        if self.ary is not None:
            self._del_desc()
            del self.ary
        self.size = -1
        self.desc = -1
    # ---

    def __del__(self) -> None:
        """Best effort finalizer"""
        try:
            self.close()
        except:  # noqa: E722
            pass

    def __len__(self) -> int:
        return len(self.ary)

    def __add__(self, other) -> "TensorArray":
        if isinstance(other, TensorArray):
            other = other.ary
        return self._view(self.ary.__add__(other))

    def __radd__(self, other) -> "TensorArray":
        if isinstance(other, TensorArray):
            other = other.ary
        return self._view(self.ary.__radd__(other))

    def __sub__(self, other) -> "TensorArray":
        if isinstance(other, TensorArray):
            other = other.ary
        return self._view(self.ary.__sub__(other))

    def __rsub__(self, other) -> "TensorArray":
        if isinstance(other, TensorArray):
            other = other.ary
        return self._view(self.ary.__rsub__(other))

    def __iadd__(self, other) -> "TensorArray":
        if isinstance(other, TensorArray):
            other = other.ary
        return self._view(self.ary.__iadd__(other))

    def __isub__(self, other) -> "TensorArray":
        if isinstance(other, TensorArray):
            other = other.ary
        return self._view(self.ary.__isub__(other))

    def __neg__(self, other) -> "TensorArray":
        if isinstance(other, TensorArray):
            other = other.ary
        return self._view(self.ary.__neg__(other))

    def __mul__(self, other) -> "TensorArray":
        if isinstance(other, TensorArray):
            other = other.ary
        return self._view(self.ary.__mul__(other))

    def __rmul__(self, other) -> "TensorArray":
        if isinstance(other, TensorArray):
            other = other.ary
        return self._view(self.ary.__rmul__(other))

    def __truediv__(self, other) -> "TensorArray":
        if isinstance(other, TensorArray):
            other = other.ary
        return self._view(self.ary.__truediv__(other))

    def __rtruediv__(self, other) -> "TensorArray":
        if isinstance(other, TensorArray):
            other = other.ary
        return self._view(self.ary.__rtruediv__(other))

    def __pow__(self, other) -> "TensorArray":
        if isinstance(other, TensorArray):
            other = other.ary
        return self._view(self.ary.__pow__(other))

    def __rpow__(self, other) -> "TensorArray":
        if isinstance(other, TensorArray):
            other = other.ary
        return self._view(self.ary.__rpow__(other))

    def __getitem__(self, index):
        return self.ary.__getitem__(index)

    def __abs__(self) -> "TensorArray":
        return self._view(self.ary.__abs__())
