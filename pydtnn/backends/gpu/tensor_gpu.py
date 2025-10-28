import ctypes
from typing import TypeVar

from enum import StrEnum, auto

import numpy as np
from pydtnn.utils.tensor import decode_tensor, TensorFormat
from pydtnn.utils.types import ArrayShape

try:
    import pycuda.gpuarray as gpuarray  # type: ignore
    from pycuda import driver as pycuda_driver  # type: ignore
    from pydtnn.backends.gpu.libs import libcudnn as cudnn
except Exception:
    pass


class TensorGPU:

    class TensorTypeEnum(StrEnum):
        TENSOR = auto()
        FILTER = auto()
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
    # ---

    def __init__(self, gpu_arr: "gpuarray.GPUArray", tensor_format: TensorFormat, cudnn_dtype: int,
                 tensor_type: TensorTypeEnum = TensorTypeEnum.TENSOR, desc: int | None = None,
                 gpudirect: bool = False, cublas: bool = False):

        self.cudnn_tensor_format = cudnn.cudnnTensorFormat['CUDNN_TENSOR_' + tensor_format.upper()]
        self.tensor_format = TensorFormat(tensor_format.lower())
        self.cudnn_dtype = cudnn_dtype
        self.tensor_type = tensor_type
        self.gpudirect = gpudirect
        self.cublas = cublas
        # The following atributes will be initalized in _initalize:
        self.ary: gpuarray.GPUArray = None
        self.size: int = -1
        self.desc: int = -1
        # ---
        self._initalize(gpu_arr, desc)
    # ---

    def _set_shape(self, gpu_arr: "gpuarray.GPUArray") -> None:

        match len(gpu_arr.shape):
            case 1:
                self.shape = (1, *gpu_arr.shape, 1, 1) if self.tensor_format is TensorFormat.NCHW else (1, 1, 1, *gpu_arr.shape)
            case 2:
                if self.tensor_format is TensorFormat.NCHW:
                    self.shape = (*gpu_arr.shape, 1, 1)
                else:
                    self.shape = (gpu_arr.shape[0], 1, 1, gpu_arr.shape[1])
            case 4:
                self.shape = gpu_arr.shape
            case _:
                raise ValueError(f"The expected len shape are 1, 2 or 4. Shape received: \"{len(gpu_arr.shape)}\".")
    # ---

    def _set_prt(self, gpu_arr: "gpuarray.GPUArray") -> None:
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
                    n, h, w, c = (self.shape[0], *decode_tensor(tuple(self.shape[1:]), tensor_format=self.tensor_format))
                    self.desc = cudnn.cudnnCreateTensorDescriptor()
                    cudnn.cudnnSetTensor4dDescriptor(self.desc, self.cudnn_tensor_format,
                                                     self.cudnn_dtype, n, c, h, w)
                case self.TensorTypeEnum.FILTER:
                    n, h, w, c = (self.shape[0], *decode_tensor(tuple(self.shape[1:]), tensor_format=self.tensor_format))
                    self.desc = cudnn.cudnnCreateFilterDescriptor()
                    cudnn.cudnnSetFilter4dDescriptor(self.desc, self.cudnn_dtype,
                                                     self.cudnn_tensor_format, n, c, h, w)
                case _:  # self.TensorTypeEnum.OTHER:
                    pass  # do nothing.
    # ---

    def _initalize(self, gpu_arr: "gpuarray.GPUArray", desc: int | None = None) -> None:
        self.ary = gpu_arr
        self._set_shape(gpu_arr)
        self.size = gpu_arr.size
        if self.size != 0:
            self._set_prt(gpu_arr)
            self._set_desc(desc)
    # ---

    def reshape(self, shape: ArrayShape):
        self.ary = self.ary.reshape(shape, order="C")
        return self
    # ---

    def free_gpu_arr(self) -> None:
        del self.ary
        self.size = -1
        self.desc = -1
    # ---

    def set_ary(self, gpu_arr: "gpuarray.GPUArray", desc: int | None = None) -> None:
        self.free_gpu_arr()
        self._initalize(gpu_arr, desc)
    # ---

    def set_ary_from_ndarray(self, arr: np.ndarray, desc: int | None = None) -> None:
        self.free_gpu_arr()
        self._initalize(gpuarray.to_gpu(arr), desc)
    # ---

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
    # ---
