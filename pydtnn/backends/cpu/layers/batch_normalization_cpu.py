
#  This file is part of Python Distributed Training of Neural Networks (PyDTNN)
#
#  Copyright (C) 2021-25 Universitat Jaume I
#
#  PyDTNN is free software: you can redistribute it and/or modify it under the
#  terms of the GNU General Public License as published by the Free Software
#  Foundation, either version 3 of the License, or (at your option) any later
#  version.
#
#  This program is distributed in the hope that it will be useful, but WITHOUT
#  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
#  or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
#  License for more details.
#
#  You should have received a copy of the GNU General Public License along
#  with this program. If not, see <https://www.gnu.org/licenses/>.
#

import numpy as np

from pydtnn.cython_modules import bn_inference_nchw_cython, bn_training_bwd_cython
from pydtnn.layers import BatchNormalization
from pydtnn.model import ModelModeEnum
from .layer_cpu import LayerCPU
from pydtnn.utils import PYDTNN_TENSOR_FORMAT
from pydtnn.utils.best_transpose_0231 import best_transpose_0231
from pydtnn.utils.best_transpose_0312 import best_transpose_0312

try:
    # noinspection PyUnresolvedReferences
    from pydtnn.libs.mpi import MPI
except (ImportError, ModuleNotFoundError):
    pass

# TODO: REVISAR: NOTE [BORRAR]: no puedes machacar "beta", "dbeta" "gamma", "dgamma"
class BatchNormalizationCPU(LayerCPU, BatchNormalization):

    def initialize(self, prev_shape):
        super().initialize(prev_shape)
        self.mu:np.ndarray = np.empty(shape=(self.ci,), dtype=self.model.dtype)
        self.var:np.ndarray = np.empty(shape=(self.ci,), dtype=self.model.dtype)
        self.dgamma:np.ndarray = np.empty(shape=(self.ci,), dtype=self.model.dtype)
        self.dbeta:np.ndarray = np.empty(shape=(self.ci,), dtype=self.model.dtype)
        self.std:np.ndarray = np.empty(shape=(self.ci,), dtype=self.model.dtype)
        if self.spatial:
            self.dx:np.ndarray = np.empty(shape=(self.model.batch_size * self.hi * self.wi, self.ci), dtype=self.model.dtype)
        else:
            # NOTE: in this case, self.hi and self.wi are 0 (self.shape should be somethin like: "(512, )"
            self.dx:np.ndarray = np.empty(shape=(self.model.batch_size, self.ci), dtype=self.model.dtype)
        
        if self.sync_stats and self.model.comm is not None and self.model.shared_storage:
            self.mean = self.mean_all_reduce
            self.n = self.model.nprocs * self.model.batch_size
        else: 
            self.mean = self.mean_numpy
            self.n = None
        
        match self.model.tensor_format:
            case PYDTNN_TENSOR_FORMAT.NCHW:
                self.y = np.empty((self.model.batch_size, self.ci, self.hi, self.wi), dtype=self.model.dtype)
            case PYDTNN_TENSOR_FORMAT.NHWC:
                self.y = np.empty((self.model.batch_size, self.hi, self.wi, self.ci), dtype=self.model.dtype)
            case _:
                raise TypeError(f"Function: \'BatchNormalizationCPU\'. Error:\n\tFormat: \'{self.model.tensor_format}\' not supported.")
    # --

    def mean_all_reduce(self, data:np.ndarray, total:int, _mean:np.ndarray) -> None:
        np.sum(data, axis=0, out=_mean)
        _mean /= total
        self.model.comm.Allreduce(MPI.IN_PLACE, _mean, op=MPI.SUM)
    # --

    def mean_numpy(self, data:np.ndarray, total:int, _mean:np.ndarray) -> None:
        np.mean(data, axis=0, out=_mean)
    # --

    def forward(self, x:np.ndarray) -> np.ndarray:

        if self.model.mode is ModelModeEnum.EVALUATE and self.spatial and self.model.tensor_format is PYDTNN_TENSOR_FORMAT.NCHW:
            y = np.zeros_like(x, order="C", dtype=x.dtype)
            bn_inference_nchw_cython(x, y, self.running_mean, self.inv_std, self.gamma, self.beta)
            return y

        if self.spatial:
            if self.model.tensor_format is PYDTNN_TENSOR_FORMAT.NCHW:
                x = best_transpose_0231(x)
            x:np.ndarray = x.reshape((-1, self.ci), copy=False)

        self.xn = x
        self.mean(self.xn, self.n, self.mu)
        self.xn -= self.mu
        #var = self.mean(xc ** 2, n, self.model.comm)        
        self.mean(self.xn ** 2, self.n, self.var)

        np.sqrt(self.var + self.epsilon, out=self.std)        
        y = self.xn / self.std
        y *= self.gamma
        y += self.beta

        if self.model.mode is ModelModeEnum.TRAIN:
            #self.running_mean = self.momentum * self.running_mean + (1.0 - self.momentum) * self.mu
            #self.mu *= (1.0 - self.momentum)
            self.running_mean *= self.momentum
            self.running_mean += (self.mu * (1.0 - self.momentum))

            #self.running_var = self.momentum * self.running_var + (1.0 - self.momentum) * self.var
            #self.var *= (1.0 - self.momentum)
            self.running_var *= self.momentum                
            self.running_var += (self.var * (1.0 - self.momentum))

        if self.spatial:
            y = y.reshape((-1, self.hi, self.wi, self.ci), copy=False)
            if self.model.tensor_format is PYDTNN_TENSOR_FORMAT.NCHW:
                y = best_transpose_0312(y)
        return y
    # --- END forward --- #

    def backward(self, dy: np.ndarray) -> np.ndarray:
        n = dy.shape[0]
        if self.spatial:
            dx:np.ndarray = self.dx[: (n * self.hi * self.wi),:]
            if self.model.tensor_format is PYDTNN_TENSOR_FORMAT.NCHW:
                dy = best_transpose_0231(dy)
            dy = dy.reshape((-1, self.ci), copy=True)
        else:
            dx:np.ndarray = self.dx[:n,:]

        np.sum(dy * self.xn, axis=0, out=self.dgamma)
        np.sum(dy, axis=0, out=self.dbeta)
        
        bn_training_bwd_cython(dx, dy, self.xn, self.std, self.gamma, self.dgamma, self.dbeta)

        if self.spatial:
            dx = dx.reshape((-1, self.hi, self.wi, self.ci), copy=False)
            if self.model.tensor_format is PYDTNN_TENSOR_FORMAT.NCHW:
                dx = best_transpose_0312(dx)
        return dx
    # --- END backward --- #
