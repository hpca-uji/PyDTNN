from pycuda import gpuarray  # type: ignore
from pydtnn.backends.pycuda.utils.tensor_array import TensorArray
from pydtnn import gpu_errors
from numpy import ndarray
from pydtnn.backends.pycuda.abstract.layerable import LayerablePycuda
from pydtnn.layers.layer import Layer
from pydtnn.utils.constants import DTYPE2CTYPE, ArrayShape
import numpy as np
import logging
logger = logging.getLogger(__name__)


class LayerPycuda(Layer[TensorArray], LayerablePycuda):
    """
    Extends a Layer class with the attributes and methods required by GPU Layers.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # GPU layer attributes
        # NOTE: All of these values will be initalized in the "initialize" method.
        self.weights_cpu: ndarray = None  # type: ignore
        self.biases_cpu: ndarray = None  # type: ignore
        self.dx: TensorArray = None  # type: ignore
        self.dw: TensorArray = None  # type: ignore
        self.db: TensorArray = None  # type: ignore
        self.dw_cpu: ndarray = None  # type: ignore
        self.db_cpu: ndarray = None  # type: ignore
        self.one_vec_cpu: ndarray = None  # type: ignore
        self.one_vec_gpu: gpuarray.GPUArray = None  # type: ignore
        self.grid: tuple[int, int, int] = None  # type: ignore
        self.block: tuple[int, int, int] = None  # type: ignore

    def _model_init(self, prev_shape: ArrayShape, x: TensorArray | None = None) -> None:
        super()._model_init(prev_shape, x)

        if not self.model.enable_cudnn:
            raise ExceptionGroup("GPU layers requires CUDNN to be enabled!", gpu_errors)

        self.grid = self.model.cuda_grid
        self.block = self.model.cuda_block

        self.defines_replaces = {
            "\"TYPE\"": DTYPE2CTYPE[self.model.dtype],
            "TENSOR_FORMAT": str(self.model.tensor_format)
        }
    # ---

    @property
    def _ary_prop(self) -> set[str]:
        return {*self.grad_vars.keys(), *self.grad_vars.values()}

    def _export_prop(self, key: str):
        if key not in self._ary_prop:
            return super()._export_prop(key)

        gpu_ary = getattr(self, key)
        cpu_ary = np.asarray(gpu_ary.get(), dtype=np.float64, order="C").copy()
        return cpu_ary

    def _import_prop(self, key: str, value) -> None:
        if key not in self._ary_prop:
            return super()._import_prop(key, value)

        gpu_ary = getattr(self, key)
        cpu_ary = np.asarray(value.reshape(gpu_ary.shape), dtype=self.model.dtype, order="C")
        gpu_ary.set(cpu_ary)
