from pydtnn.utils.constants import DTYPE2CTYPE, ArrayShape
from typing import TYPE_CHECKING
from pydtnn.libs import numpy as np
from pydtnn.backends.numpy.layers.batch_normalization import BatchNormalizationNumpy
from pydtnn.backends.cupy.layers.layer import LayerCupy
import logging
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy as np


class BatchNormalizationCupy(BatchNormalizationNumpy, LayerCupy):

    def _model_init(self, prev_shape: ArrayShape, x: np.ndarray | None = None) -> None:
        super()._model_init(prev_shape, x)

        self.stream_2 = np.cuda.Stream()

        self.bwd = self._bwd_kernel()
        # ----

    def _training_bwd(self, dx: np.ndarray, dy: np.ndarray) -> None:
        # return super()._training_bwd(dx, dy)
        dim_i, dim_j = dx.shape
        self.bwd(self.model.cuda_grid,
                 self.model.cuda_block,
                 (dx, dy, self.xn,
                  self.std, self.gamma,
                  self.dgamma, self.dbeta,
                  dim_i, dim_j, dx.size))
    # ---

    def _bwd_kernel(self, func_name: str = "bn_training_bwd") -> np.RawKernel:

        code = \
            r"""
extern "C"
#define GET_J(idx, dim_j) (idx % dim_j)
#define INDEX_FIRST_ELEMENT(index, dim_in, dim_out) ((index * dim_in) / dim_out)
#define INDEX_LAST_ELEMENT(index, dim_in, dim_out) ((((index + 1) * dim_in) + dim_out - 1) / dim_out)
#define IS_BETWEEN(min_v, var, max_v) (min_v <= var) && (var < max_v)

__global__ void {FUNC_NAME}({T}* dx, {T}* dy, {T}* xn,
                            {T}* std, {T}* gamma,
                            {T}* dgamma, {T}* dbeta,
                            int dim_i, int dim_j, int N)
{{
    int j;
    const int n = (const int) dim_i;
    {T} _gamma, _std, _dy, _xn, _dgamma, _dbeta;

    // BLOCK DISTRIBUTION
    int idx;
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
    // BLOCK DISTRIBUTION

    for(idx = n_offset; idx < end_offset; idx++)
    {{
        j = GET_J(idx, dim_j);

        _gamma = *(gamma + j);      // gamma[j]
        _std = *(std + j);          // std[j]
        _dy = *(dy + idx);          // dy[idx] = dy[i][j]
        _xn = *(xn + idx);          // xn[idx] = xn[i][j]
        _dgamma = *(dgamma + j);    // dgamma[j]
        _dbeta = *(dbeta + j);      // dbeta[j]

        *(dx + idx) = ({T}) ( _gamma / ( _std * n) ) * (n * _dy - _xn * _dgamma - _dbeta);
    }}
}}
"""
        code = code.format(FUNC_NAME=func_name, T=DTYPE2CTYPE[self.model.dtype])
        return np.RawKernel(code, func_name, backend=self.cuda_compiler)
    # ---
