from pydtnn.backends.numpy.activations.leaky_relu import LeakyReluNumpy
from pydtnn.backends.cupy.activations.activation import ActivationCupy
from pydtnn.utils.constants import DTYPE2CTYPE
from pydtnn.libs import numpy as np


class LeakyReluCupy(LeakyReluNumpy, ActivationCupy):

    def _model_init(self, prev_shape, x=None):
        super()._model_init(prev_shape, x)
        self.fwd = self.leaky_relu_fwd()
        self.bwd = self.leaky_relu_bwd()


    def forward(self, x: np.ndarray) -> np.ndarray:
        self.y = self._y[:x.shape[0], :]
        self.mask = self._mask[:x.shape[0], :]

        self.fwd(self.model.cuda_grid,
                 self.model.cuda_block,
                 (x, self.y, self.mask, self.negative_slope, x.size))
        return self.y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        self.bwd(self.model.cuda_grid,
                 self.model.cuda_block,
                 (dy, dy, self.mask, self.negative_slope, dy.size))
        return dy

    def leaky_relu_fwd(self) -> np.RawKernel:
        func_name = "leaky_relu_fwd"
        code = \
            r"""
extern "C"
__global__ void {FUNC_NAME}({T}* x, {T}* max, {T}* mask,
                            float negative_slope, int N)
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

        if (elem > 0)
        {{
            max[i] = elem;
            mask[i] = 1;
        }}
        else if(elem < 0)
        {{
            max[i] = ({T}) (elem * negative_slope);
            mask[i] = negative_slope;
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

    def leaky_relu_bwd(self) -> np.RawKernel:
        func_name = "leaky_relu_bwd"
        code = \
            r"""
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
