
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

from pydtnn.cython_modules import bn_inference_cython, bn_inference_nchw_cython, bn_training_bwd_cython
from pydtnn.layers import BatchNormalization
from pydtnn.model import ModelModeEnum
from .layer_cpu import LayerCPU
from pydtnn.utils import PYDTNN_TENSOR_FORMAT

try:
    # noinspection PyUnresolvedReferences
    from pydtnn.libs.mpi import MPI
except (ImportError, ModuleNotFoundError):
    pass


class BatchNormalizationCPU(LayerCPU, BatchNormalization):

    def initialize(self, prev_shape):
        super().initialize(prev_shape)
        self.mu:np.ndarray = np.empty(shape=(self.ci,), dtype=self.model.dtype)
        self.var:np.ndarray = np.empty(shape=(self.ci,), dtype=self.model.dtype)
        self.dgamma:np.ndarray = np.empty(shape=(self.ci,), dtype=self.model.dtype)
        self.dbeta:np.ndarray = np.empty(shape=(self.ci,), dtype=self.model.dtype)
        self.std:np.ndarray = np.empty(shape=(self.ci,), dtype=self.model.dtype)
        
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

        if self.model.mode == ModelModeEnum.EVALUATE and self.spatial and self.model.tensor_format == PYDTNN_TENSOR_FORMAT.NCHW:
            y = np.zeros_like(x, order="C", dtype=x.dtype)
            bn_inference_nchw_cython(x, y, self.running_mean, self.inv_std, self.gamma, self.beta)
            return y

        if self.spatial:
            x:np.ndarray = x.reshape((-1, self.ci), copy=False)

        self.xn = x
        self.mean(self.xn, self.n, self.mu)
        self.xn -= self.mu
        #var = self.mean(xc ** 2, n, self.model.comm)
        self.mean(self.xn ** 2, self.n, self.var)

        self.std = np.sqrt(self.var + self.epsilon)
        self.xn /= self.std
        y = self.gamma * self.xn 
        y += self.beta

        if self.model.mode is ModelModeEnum.TRAIN:
            #self.running_mean = self.momentum * self.running_mean + (1.0 - self.momentum) * self.mu
            self.mu *= (1.0 - self.momentum)
            self.running_mean *= self.momentum
            self.running_mean += self.mu

            #self.running_var = self.momentum * self.running_var + (1.0 - self.momentum) * self.var
            self.var *= (1.0 - self.momentum)                
            self.running_var *= self.momentum                
            self.running_var += self.var

        if self.spatial:                
            match self.model.tensor_format:
                case PYDTNN_TENSOR_FORMAT.NHWC:
                    y = y.reshape((-1, self.hi, self.wi, self.ci), copy=False)
                case PYDTNN_TENSOR_FORMAT.NCHW:
                    y = y.reshape((-1, self.ci, self.hi, self.wi), copy=False)
                case _:
                    raise NotImplementedError(f"Operation not implemented in \'{self.model.tensor_format}\' format")
        return y
    # --- END forward --- #

    def backward(self, dy: np.ndarray) -> np.ndarray | None:
        if self.spatial:
            dy = dy.reshape((-1, self.ci), copy=True)

        np.sum(dy * self.xn, axis=0, out=self.dgamma)
        np.sum(dy, axis=0, out=self.dbeta)

        # dx = (self.gamma / (self.std * n)) * (n * dy - self.xn * self.dgamma - self.dbeta)
        # dx = dx.astype(self.model.dtype)
        dx:np.ndarray = np.empty(shape=dy.shape, dtype=self.model.dtype, order="C")
        
        bn_training_bwd_cython(dx, dy, self.xn, self.std, self.gamma, self.dgamma, self.dbeta)

        if self.spatial:
            match self.model.tensor_format:
                case PYDTNN_TENSOR_FORMAT.NHWC:
                    dx = dx.reshape((-1, self.hi, self.wi, self.ci), copy=False)
                case PYDTNN_TENSOR_FORMAT.NCHW:
                    dx = dx.reshape((-1, self.ci, self.hi, self.wi), copy=False)
                case _:
                    raise NotImplementedError(f"Operation not implemented in \'{self.model.tensor_format}\' format")

        return dx
    # --- END backward --- #
