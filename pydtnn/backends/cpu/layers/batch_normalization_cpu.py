
#  This file is part of Python Distributed Training of Neural Networks (PyDTNN)
#
#  Copyright (C) 2021-22 Universitat Jaume I
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

from pydtnn.cython_modules import bn_inference_cython, bn_inference_nchw_cython, bn_training_fwd_cython, \
                                  bn_training_bwd_cython
from pydtnn.layers import BatchNormalization
from pydtnn.model import ModelModeEnum
from pydtnn.utils.best_transpose_0231 import best_transpose_0231
from pydtnn.utils.best_transpose_0312 import best_transpose_0312
from .layer_cpu import LayerCPU
from pydtnn.utils import PYDTNN_TENSOR_FORMAT

try:
    # noinspection PyUnresolvedReferences
    from pydtnn.libs.mpi import MPI
except (ImportError, ModuleNotFoundError):
    pass


class BatchNormalizationCPU(LayerCPU, BatchNormalization):

    def forward(self, x:np.ndarray) -> np.ndarray:

        def mean(data:np.ndarray, total:int, comm) -> np.ndarray:
            if self.sync_stats and comm is not None and self.model.shared_storage:
                _mean:np.ndarray = np.sum(data, axis=0) / total
                comm.Allreduce(MPI.IN_PLACE, _mean, op=MPI.SUM)
            else:
                _mean:np.ndarray = np.mean(data, axis=0)
            return _mean
        # -- End mean -- #

        if self.model.mode == ModelModeEnum.EVALUATE and self.spatial and self.model.tensor_format == PYDTNN_TENSOR_FORMAT.NCHW:
            y:np.ndarray = bn_inference_nchw_cython(x, self.running_mean, self.inv_std, self.gamma, self.beta)
            return y

        if self.spatial:
            if self.model.tensor_format == PYDTNN_TENSOR_FORMAT.NCHW:
                x:np.ndarray = best_transpose_0231(x)
            x:np.ndarray = x.reshape((-1, self.ci), copy=False)

        match self.model.mode:
            case ModelModeEnum.TRAIN:
                if self.sync_stats and self.model.comm is not None and self.model.shared_storage:
                    n = self.model.nprocs * self.model.batch_size
                    # n = np.array([x.shape[0]], dtype=self.model.dtype)
                    # self.model.comm.Allreduce(MPI.IN_PLACE, n, op=MPI.SUM)
                else: 
                    n = None

                mu = mean(x, n, self.model.comm)
                xc = (x - mu)
                var = mean(xc ** 2, n, self.model.comm)

                self.std:np.ndarray = np.sqrt(var + self.epsilon)
                self.xn:np.ndarray = xc / self.std
                y:np.ndarray = self.gamma * self.xn
                y += self.beta

                #self.running_mean = self.momentum * self.running_mean + (1.0 - self.momentum) * mu
                mu *= (1.0 - self.momentum)
                self.running_mean *= self.momentum
                self.running_mean += mu

                #self.running_var = self.momentum * self.running_var + (1.0 - self.momentum) * var
                var *= (1.0 - self.momentum)                
                self.running_var *= self.momentum                
                self.running_var += var

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
                    self.inv_std = 1.0 / np.sqrt(self.running_var + self.epsilon)

                y = bn_inference_cython(x, self.running_mean, self.inv_std, self.gamma, self.beta)                
            case _:
                raise RuntimeError(f"Unexpected model mode '{self.model.mode}'.")

        if self.spatial:
            y = y.reshape((-1, self.hi, self.wi, self.ci), copy=False)
            if self.model.tensor_format == PYDTNN_TENSOR_FORMAT.NCHW:
                y = best_transpose_0312(y)

        return y
    # --- END forward --- #

    def backward(self, dy: np.ndarray) -> np.ndarray | None:
        if self.spatial:
            if self.model.tensor_format == PYDTNN_TENSOR_FORMAT.NCHW:
                dy = best_transpose_0231(dy)
            dy = dy.reshape((-1, self.ci), copy=False)

        self.dgamma = np.sum(dy * self.xn, axis=0)
        self.dbeta = np.sum(dy, axis=0)

        if self.need_dx:
            # dx = (self.gamma / (self.std * n)) * (n * dy - self.xn * self.dgamma - self.dbeta)
            # dx = dx.astype(self.model.dtype)

            dx:np.ndarray = bn_training_bwd_cython(dy, self.std, self.xn, self.gamma, self.dgamma, self.dbeta)

            if self.spatial:
                dx = dx.reshape((-1, self.hi, self.wi, self.ci), copy=False)
                if self.model.tensor_format == PYDTNN_TENSOR_FORMAT.NCHW:
                    dx = best_transpose_0312(dx)

            return dx
    # --- END backward --- #
