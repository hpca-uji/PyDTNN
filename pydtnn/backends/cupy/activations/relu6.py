import logging
logger = logging.getLogger(__name__)

from pydtnn.libs import numpy as np
from pydtnn.backends.numpy.activations.relu6 import Relu6Numpy
from pydtnn.backends.cupy.activations.activation import ActivationCupy
from pydtnn.utils.constants import DTYPE2CTYPE

class Relu6Cupy(Relu6Numpy, ActivationCupy):

    def _model_init(self, prev_shape, x=None):
        super()._model_init(prev_shape, x)
        self.fwd = self.relu6_fwd()
        self.bwd = self.relu6_bwd()


    def forward(self, x: np.ndarray) -> np.ndarray:
        self.y = np.ascontiguousarray(self._y[:x.shape[0], :], dtype=self.model.dtype)
        self.mask = np.ascontiguousarray(self._mask[:x.shape[0], :], dtype=self.model.dtype)

        self.fwd(self.model.cuda_grid,
                 self.model.cuda_block,
                 (x, self.y, self.mask, self.cap, x.size))
        return self.y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        self.bwd(self.model.cuda_grid,
                 self.model.cuda_block,
                 (dy, dy, self.mask, dy.size))
        return dy

    def relu6_fwd(self) -> np.RawKernel:
        func_name = "relu6_fwd"
        code = \
            r"""
extern "C"
__global__ void {FUNC_NAME}({T}* x, {T}* max, {T}* mask,
                            float cap, int N)
{{
    int i;
    {T} elem;

    const int base_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_workers = blockDim.x * gridDim.x;
    int samples_worker, samples_overworker, overworkers;
    int n_samples, n_offset, end_offset;

    overworkers = N % num_workers;
    samples_worker = N / num_workers;
    samples_overworker = samples_worker + 1;

    if (base_idx < overworkers)
    {{
        n_samples = samples_overworker;
        n_offset = base_idx * n_samples;
    }}
    else
    {{
        n_samples = samples_worker;
        n_offset = samples_overworker * overworkers + n_samples * (base_idx - overworkers);
    }}
    end_offset = n_offset + n_samples;

    for(i = n_offset; i < end_offset; i++)
    {{
        elem = x[i];

        if (elem >= cap)
        {{
            max[i] = cap;
            mask[i] = 1;
        }}
        else if (elem > 0)
        {{
            max[i] = elem;
            mask[i] = 1;
        }}
        else
        {{
            max[i] = 0;
            mask[i] = 0;
        }}
    }}
}}
"""
        code = code.format(FUNC_NAME=func_name, T=DTYPE2CTYPE[self.model.dtype])

        return np.RawKernel(code, func_name, backend=self.cuda_compiler)
    # -----

    def relu6_bwd(self) -> np.RawKernel:
        func_name = "relu6_bwd"
        code = \
            r"""
extern "C"
__global__ void {FUNC_NAME}({T}* dx, {T}* dy, {T}* mask, int N)
{{
    int i;
    {T} elem;

    const int base_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_workers = blockDim.x * gridDim.x;
    int samples_worker, samples_overworker, overworkers;
    int n_samples, n_offset, end_offset;

    overworkers = N % num_workers;
    samples_worker = N / num_workers;
    samples_overworker = samples_worker + 1;

    if (base_idx < overworkers)
    {{
        n_samples = samples_overworker;
        n_offset = base_idx * n_samples;
    }}
    else
    {{
        n_samples = samples_worker;
        n_offset = samples_overworker * overworkers + n_samples * (base_idx - overworkers);
    }}
    end_offset = n_offset + n_samples;

    for(i = n_offset; i < end_offset; i++)
        dx[i] = dy[i] * mask[i];
}}
"""
        code = code.format(FUNC_NAME=func_name, T=DTYPE2CTYPE[self.model.dtype])

        return np.RawKernel(code, func_name, backend=self.cuda_compiler)
    # -----

