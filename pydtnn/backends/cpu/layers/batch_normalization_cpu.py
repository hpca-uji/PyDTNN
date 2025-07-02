
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

    def initialize(self, prev_shape, need_dx=True):
        super().initialize(prev_shape, need_dx)
        self.mu:np.ndarray = np.empty(shape=(self.ci,), dtype=self.model.dtype)
        self.var:np.ndarray = np.empty(shape=(self.ci,), dtype=self.model.dtype)
        self.dgamma:np.ndarray = np.empty(shape=(self.ci,), dtype=self.model.dtype)
        self.dbeta:np.ndarray = np.empty(shape=(self.ci,), dtype=self.model.dtype)
        
        if self.sync_stats and self.model.comm is not None and self.model.shared_storage:
            self.mean = self.mean_all_reduce
        else: 
            self.mean = self.mean_numpy
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

        match self.model.mode:
            case ModelModeEnum.TRAIN:
                if self.sync_stats and self.model.comm is not None and self.model.shared_storage:
                    n = self.model.nprocs * self.model.batch_size
                    # n = np.array([x.shape[0]], dtype=self.model.dtype)
                    # self.model.comm.Allreduce(MPI.IN_PLACE, n, op=MPI.SUM)
                else: 
                    n = None

                self.mean(x, n, self.mu)
                x -= self.mu
                x *= x                
                self.mean(x, n, self.var)                
                self.xn:np.ndarray = x
                

                self.std:np.ndarray = np.sqrt(self.var + self.epsilon)
                self.xn /= self.std
                y:np.ndarray = self.gamma * self.xn
                y += self.beta

                #self.running_mean = self.momentum * self.running_mean + (1.0 - self.momentum) * mu
                self.mu *= (1.0 - self.momentum)
                self.running_mean *= self.momentum
                self.running_mean += self.mu

                #self.running_var = self.momentum * self.running_var + (1.0 - self.momentum) * var
                self.var *= (1.0 - self.momentum)                
                self.running_var *= self.momentum                
                self.running_var += self.var

                # y, self.std, self.xn = bn_training_fwd_cython(x, self.gamma, self.beta, \
                #                                               self.running_mean, self.running_var, \
                #                                               self.momentum, self.epsilon)

                self.updated_running_var = True

            case ModelModeEnum.EVALUATE:
                # Original numpy-based code
                # std = np.sqrt(self.running_var + self.epsilon)
                # xn = (x - self.running_mean) / std
                # y = self.gamma * xn + self.beta

                # If self.running_var was updated on training we need to recompute self.inv_std!

                if self.updated_running_var:
                    self.updated_running_var = False
                    # self.inv_std = 1.0 / np.sqrt(self.running_var + self.epsilon)
                    self.inv_std = self.running_var + self.epsilon
                    np.sqrt(self.inv_std, out=self.inv_std)
                    np.reciprocal(self.inv_std, out=self.inv_std)                    

                y = np.empty_like(x, order="C", dtype=self.model.dtype)
                bn_inference_cython(x, y, self.running_mean, self.inv_std, self.gamma, self.beta)                
            case _:
                raise RuntimeError(f"Unexpected model mode '{self.model.mode}'.")

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
            dy = dy.reshape((-1, self.ci), copy=False)

        np.sum(dy * self.xn, axis=0, out=self.dgamma)
        np.sum(dy, axis=0, out=self.dbeta)

        if self.need_dx:
            # dx = (self.gamma / (self.std * n)) * (n * dy - self.xn * self.dgamma - self.dbeta)
            # dx = dx.astype(self.model.dtype)
            dx:np.ndarray = np.empty(shape=(self.wi * self.hi * dy.shape[0], self.ci), dtype=self.model.dtype, order="C")

            bn_training_bwd_cython(dx, dy, self.xn, self.std, self.gamma, self.dgamma, self.dbeta)

            match self.model.tensor_format:
                case PYDTNN_TENSOR_FORMAT.NHWC:
                    dx = dx.reshape((-1, self.hi, self.wi, self.ci), copy=False)
                case PYDTNN_TENSOR_FORMAT.NCHW:
                    dx = dx.reshape((-1, self.ci, self.hi, self.wi), copy=False)
                case _:
                    raise NotImplementedError(f"Operation not implemented in \'{self.model.tensor_format}\' format")

            return dx
    # --- END backward --- #
