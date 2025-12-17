import ctypes
import copy
from enum import StrEnum, auto

import numpy as np

from pydtnn.utils.constants import ArrayShape
from pydtnn.utils.tensor import TensorFormat, encode_shape, decode_shape

try:
    import pycuda.gpuarray as gpuarray  # type: ignore
    from pycuda import driver as pycuda_driver  # type: ignore
    from pydtnn.libs import libcudnn as cudnn
except Exception:
    pass


class TensorGPU:

    class TensorTypeEnum(StrEnum):
        TENSOR = auto()
        FILTER = auto()
        SEQ = auto()
        OTHER = auto()
    # ---  END EnumTensorType --- #

    @staticmethod
    def create_empty_tensor(shape: ArrayShape, dtype: np.dtype,
                            tensor_format: TensorFormat, cudnn_dtype: int,
                            tensor_type: TensorTypeEnum = TensorTypeEnum.TENSOR, desc: int | None = None,
                            gpudirect: bool = False, cublas: bool = False):
        gpu_arr = gpuarray.empty(shape, dtype)
        return TensorGPU(gpu_arr=gpu_arr, tensor_format=tensor_format, cudnn_dtype=cudnn_dtype,
                         tensor_type=tensor_type, desc=desc, gpudirect=gpudirect, cublas=cublas)
    # ---

    @staticmethod
    def create_zeros_tensor(shape: ArrayShape, dtype: np.dtype,
                            tensor_format: TensorFormat, cudnn_dtype: int,
                            tensor_type: TensorTypeEnum = TensorTypeEnum.TENSOR, desc: int | None = None,
                            gpudirect: bool = False, cublas: bool = False):
        gpu_arr = gpuarray.zeros(shape, dtype)
        return TensorGPU(gpu_arr=gpu_arr, tensor_format=tensor_format, cudnn_dtype=cudnn_dtype,
                         tensor_type=tensor_type, desc=desc, gpudirect=gpudirect, cublas=cublas)
    
    @staticmethod
    def initialize_gpu_direct(drv: "pycuda_driver", shape: ArrayShape, dtype: np.dtype,
                              tensor_format: TensorFormat, cudnn_dtype: int,
                              tensor_type: TensorTypeEnum = TensorTypeEnum.TENSOR,
                              desc: int | None = None, gpudirect: bool = False, cublas: bool = False) -> tuple[np.ndarray, "TensorGPU"]:
        x_cpu = drv.aligned_zeros(shape, dtype)
        x_gpu = drv.register_host_memory(x_cpu, flags=drv.mem_host_register_flags.DEVICEMAP)

        x_gpu = TensorGPU(x_gpu, tensor_format=tensor_format, cudnn_dtype=cudnn_dtype, tensor_type=tensor_type,
                          desc=desc, gpudirect=gpudirect, cublas=cublas)

        return (x_cpu, x_gpu)
    # ---

    @staticmethod
    def initialize_not_gpu_direct(shape: ArrayShape, dtype: np.dtype,
                                  tensor_format: TensorFormat, cudnn_dtype: int,
                                  tensor_type: TensorTypeEnum = TensorTypeEnum.TENSOR,
                                  desc: int | None = None, gpudirect: bool = False, cublas: bool = False) -> tuple[np.ndarray, "TensorGPU"]:
        x_cpu = np.zeros(shape, dtype)
        x_gpu = gpuarray.empty(shape, dtype)

        x_gpu = TensorGPU(x_gpu, tensor_format=tensor_format, cudnn_dtype=cudnn_dtype, tensor_type=tensor_type,
                          desc=desc, gpudirect=gpudirect, cublas=cublas)

        return (x_cpu, x_gpu)
    
    @staticmethod
    def initialize(shape: ArrayShape, dtype: np.dtype,
                   tensor_format: TensorFormat, cudnn_dtype: int,
                   tensor_type: TensorTypeEnum = TensorTypeEnum.TENSOR,
                   desc: int | None = None, gpudirect: bool = False, cublas: bool = False,
                   drv: "pycuda_driver" = None) -> tuple[np.ndarray, "TensorGPU"]:
        if drv is not None:
            return TensorGPU.initialize_gpu_direct(drv=drv, shape=shape,
                                                   dtype=dtype, tensor_format=tensor_format,
                                                   cudnn_dtype=cudnn_dtype, tensor_type=tensor_type,
                                                   desc=desc, gpudirect=gpudirect, cublas=cublas)
        else:
            return TensorGPU.initialize_not_gpu_direct(shape=shape, dtype=dtype, tensor_format=tensor_format,
                                                       cudnn_dtype=cudnn_dtype, tensor_type=tensor_type,
                                                       desc=desc, gpudirect=gpudirect, cublas=cublas)


    # ---

    def __init__(self, gpu_arr: "gpuarray.GPUArray", tensor_format: TensorFormat, cudnn_dtype: int,
                 tensor_type: TensorTypeEnum = TensorTypeEnum.TENSOR, desc: int | None = None,
                 gpudirect: bool = False, cublas: bool = False):

        self.tensor_format = TensorFormat(tensor_format.lower())
        self.cudnn_dtype = cudnn_dtype
        self.tensor_type = tensor_type
        self.gpudirect = gpudirect
        self.cublas = cublas
        self.cudnn_tensor_format = cudnn.cudnnTensorFormat['CUDNN_TENSOR_' + tensor_format.upper()]
        # The following atributes will be initalized in _initalize:
        self.ary: gpuarray.GPUArray = None
        self.size: int = -1
        self.desc: int = -1
        # ---
        self._initalize(gpu_arr, desc)
    # ---

    def copy(self):
        """ NumPy-like copy. """
        return copy.deepcopy(self)

    def __copy__(self):
        return TensorGPU(gpu_arr=self.ary,
                         tensor_format=self.tensor_format,
                         cudnn_dtype=self.cudnn_dtype,
                         tensor_type=self.tensor_type,
                         gpudirect=self.gpudirect,
                         cublas=self.cublas,
                         desc=self.desc)

    def __deepcopy__(self, memo: dict):
        obj = TensorGPU(gpu_arr=copy.deepcopy(self.ary, memo),
                        tensor_format=self.tensor_format,
                        cudnn_dtype=self.cudnn_dtype,
                        tensor_type=self.tensor_type,
                        gpudirect=self.gpudirect,
                        cublas=self.cublas,
                        desc=-1)
        memo[id(self)] = obj
        return obj

    def encode_shape(self, shape):
        return encode_shape(shape, self.tensor_format)

    def decode_shape(self, shape):
        return decode_shape(shape, self.tensor_format)

    def _set_shape(self, gpu_arr: "gpuarray.GPUArray") -> None:

        match len(gpu_arr.shape):
            case 1:
                match self.tensor_format:
                    case TensorFormat.NCHW:
                        self.shape = (1, *gpu_arr.shape, 1, 1)
                    case TensorFormat.NHWC:
                        self.shape = (1, 1, 1, *gpu_arr.shape)
                    case tensor_format:
                        raise NotImplementedError(f"Unsupported tensor format {tensor_format}!")
            case 2:
                match self.tensor_format:
                    case TensorFormat.NCHW:
                        self.shape = (*gpu_arr.shape, 1, 1)
                    case TensorFormat.NHWC:
                        self.shape = (gpu_arr.shape[0], 1, 1, gpu_arr.shape[1])
                    case tensor_format:
                        raise NotImplementedError(f"Unsupported tensor format {tensor_format}!")
            case 3:
                match self.tensor_format:
                    case TensorFormat.NCHW:
                        self.shape = (gpu_arr.shape[0], 1, gpu_arr.shape[1], gpu_arr.shape[2])
                    case TensorFormat.NHWC:
                        raise NotImplementedError("Shape padding not implemented for 3-dim shape on NHWC")
            case 4:
                self.shape = gpu_arr.shape
            case _:
                raise ValueError(f"The expected len shape are 1, 2 or 4. Shape received: {len(gpu_arr.shape)}.")
    # ---

    def _set_ptr(self, gpu_arr: "gpuarray.GPUArray") -> None:
        if self.gpudirect:
            self.ptr_intp = np.intp(self.ary.base.get_device_pointer())
            self.ptr = ctypes.c_void_p(int(self.ary.base.get_device_pointer()))
        else:
            self.ptr = ctypes.c_void_p(int(gpu_arr.gpudata))
    # ---

    def _set_desc(self, desc: int | None) -> None:
        if desc is not None:
            self.desc = desc
        else:
            match self.tensor_type:
                case self.TensorTypeEnum.TENSOR:
                    n, c, h, w = self.decode_shape(self.shape)
                    self.desc = cudnn.cudnnCreateTensorDescriptor()
                    cudnn.cudnnSetTensor4dDescriptor(self.desc, self.cudnn_tensor_format,
                                                     self.cudnn_dtype, n, c, h, w)
                case self.TensorTypeEnum.FILTER:
                    n, c, h, w = self.decode_shape(self.shape)
                    self.desc = cudnn.cudnnCreateFilterDescriptor()
                    cudnn.cudnnSetFilter4dDescriptor(self.desc, self.cudnn_dtype,
                                                     self.cudnn_tensor_format, n, c, h, w)
                case self.TensorTypeEnum.SEQ:
                    if len(self.shape) == 3:
                        self.shape = (self.shape[0], 1, self.shape[-2], self.shape[-1])
                    self.desc = cudnn.cudnnCreateSeqDataDescriptor()
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
                    cudnn.cudnnSetSeqDataDescriptor(self.desc, cudnn_dtype,
                                                    np.int32(4), dimA, axes,
                                                    np.int32(len(self.seq_length_array)), self.seq_length_array,
                                                    None)
                case self.TensorTypeEnum.OTHER:
                    pass  # do nothing.

                case tensor_type:
                    raise NotImplementedError(f"Tensor type not implemented! ({tensor_type})")
    # ---

    def _del_desc(self) -> None:
        match self.tensor_type:
            case self.TensorTypeEnum.TENSOR:
                cudnn.cudnnDestroyTensorDescriptor(self.desc)
            case self.TensorTypeEnum.FILTER:
                cudnn.cudnnDestroyFilterDescriptor(self.desc)
            case self.TensorTypeEnum.SEQ:
                pass
            case self.TensorTypeEnum.OTHER:
                cudnn.cudnnDestroySeqDataDescriptor(self.desc)
            case tensor_type:
                raise NotImplementedError(f"Tensor type not implemented! ({tensor_type})")
        self.desc = -1

    def _initalize(self, gpu_arr: "gpuarray.GPUArray", desc: int | None = None) -> None:
        self.ary = gpu_arr
        self._set_shape(gpu_arr)
        self.size = gpu_arr.size
        if self.size != 0:
            self._set_ptr(gpu_arr)
            self._set_desc(desc)
    # ---

    def reshape(self, shape: ArrayShape):
        self.ary = self.ary.reshape(shape, order="C")
        self.shape = shape
        return self
    # ---

    def free_gpu_arr(self) -> None:
        if self.ary is not None:
            self._del_desc()
            del self.ary
        self.size = -1
        self.desc = -1
    # ---

    def __del__(self) -> None:
        """Best effort finalizer"""
        try:
            self.free_gpu_arr()
        except:  # noqa: E722
            pass

    def set_ary(self, gpu_arr: "gpuarray.GPUArray", desc: int | None = None) -> None:
        self.free_gpu_arr()
        self._initalize(gpu_arr, desc)
    # ---

    def set_ary_from_ndarray(self, arr: np.ndarray, desc: int | None = None) -> None:
        self.free_gpu_arr()
        self._initalize(gpuarray.to_gpu(arr), desc)
    # ---

    def fill(self, scalar: int | float) -> None:
        self.ary.fill(scalar)
    # ---
