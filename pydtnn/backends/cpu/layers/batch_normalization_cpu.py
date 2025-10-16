import numpy as np

from pydtnn.cython_modules import bn_training_bwd_cython
from pydtnn.layers import BatchNormalization
from pydtnn.model import Model
from pydtnn.backends.cpu.layers.layer_cpu import LayerCPU
from pydtnn.utils.tensor import PYDTNN_TENSOR_FORMAT
from pydtnn.utils.best_transpose_0231 import best_transpose_0231
from pydtnn.utils.best_transpose_0312 import best_transpose_0312

try:
    # noinspection PyUnresolvedReferences
    from pydtnn.comm import MPI
except Exception as e:
    pass


class BatchNormalizationCPU(LayerCPU, BatchNormalization[np.ndarray]):

    def initialize(self, prev_shape, x = None):
        super().initialize(prev_shape, x)
        self.mu: np.ndarray = np.empty(shape=(self.ci,), dtype=self.model.dtype, order="C")
        self.mu_var_momentum: np.ndarray = np.empty(shape=(self.ci,), dtype=self.model.dtype, order="C")
        self.var: np.ndarray = np.empty(shape=(self.ci,), dtype=self.model.dtype, order="C")
        self.var_eps: np.ndarray = np.empty(shape=(self.ci,), dtype=self.model.dtype, order="C")
        self.dgamma: np.ndarray = np.empty(shape=(self.ci,), dtype=self.model.dtype, order="C")
        self.dbeta: np.ndarray = np.empty(shape=(self.ci,), dtype=self.model.dtype, order="C")
        self.std: np.ndarray = np.empty(shape=(self.ci,), dtype=self.model.dtype, order="C")
        if self.spatial:
            self.dx: np.ndarray = np.empty(shape=(self.model.batch_size * self.hi * self.wi, self.ci), dtype=self.model.dtype, order="C")
            self.y = np.empty((self.model.batch_size * self.hi * self.wi, self.ci), dtype=self.model.dtype, order="C")
            self.dy_xn = np.empty((self.model.batch_size * self.hi * self.wi, self.ci), dtype=self.model.dtype, order="C")
        else:
            # NOTE: in this case, self.hi and self.wi are 0 (self.shape should be somethin like: "(512, )"
            self.dx: np.ndarray = np.empty(shape=(self.model.batch_size, self.ci), dtype=self.model.dtype, order="C")
            self.y = np.empty((self.model.batch_size, self.ci), dtype=self.model.dtype, order="C")
            self.dy_xn = np.empty((self.model.batch_size, self.ci), dtype=self.model.dtype, order="C")
            
    # --

    def forward(self, x: np.ndarray) -> np.ndarray:

        if self.spatial:
            x: np.ndarray = x.reshape((-1, self.ci), copy=False, order="C")

        y: np.ndarray = self.y[:x.shape[0], :]

        if self.model.mode is Model.Mode.EVALUATE:
            # y = self.gamma * (x - self.running_mean) / np.sqrt(self.running_var + self.epsilon) + self.beta

            np.subtract(x, self.running_mean, out=x, 
                        dtype=self.model.dtype)
            np.multiply(self.gamma, x, out=y, 
                        dtype=self.model.dtype)
            np.sqrt(self.running_var + self.epsilon, out=self.std, 
                        dtype=self.model.dtype)
            np.divide(y, self.std, out=y, 
                        dtype=self.model.dtype)
            np.add(y, self.beta, out=y, 
                        dtype=self.model.dtype)

        else:  # ModelModeEnum.TRAIN:
            
            # NOTE: casting="unsafe", means that numpy will cast the data to the new type always.
            self.xn = x

            np.mean(self.xn, axis=0, out=self.mu, 
                    dtype=self.model.dtype)
            np.subtract(self.xn, self.mu, out=self.xn,
                        dtype=self.model.dtype)
            np.mean(self.xn ** 2, axis=0, out=self.var, 
                    dtype=self.model.dtype)

            np.add(self.var, self.epsilon, out=self.var_eps, 
                   dtype=self.model.dtype)
            np.sqrt(self.var_eps + self.epsilon, out=self.std, 
                      order="C", dtype=self.model.dtype)
            np.divide(self.xn, self.std, out=self.xn, 
                      order="C", dtype=self.model.dtype)

            np.multiply(self.gamma, self.xn, out=y, 
                        dtype=self.model.dtype)

            np.add(y, self.beta, out=y, 
                   dtype=self.model.dtype)

            # self.running_mean = self.momentum * self.running_mean + (1.0 - self.momentum) * self.mu
            inv_momentum = (1.0 - self.momentum)

            np.multiply(self.running_mean, self.momentum, out=self.running_mean, 
                        dtype=self.model.dtype)
            np.multiply(self.mu, inv_momentum, out=self.mu_var_momentum, 
                        dtype=self.model.dtype)
            np.add(self.running_mean, self.mu_var_momentum, out=self.running_mean, 
                   order="C", dtype=self.model.dtype, casting="unsafe")

            # self.running_var = self.momentum * self.running_var + (1.0 - self.momentum) * self.var
            np.multiply(self.running_var, self.momentum, out=self.running_var, 
                        dtype=self.model.dtype)
            np.multiply(self.var, inv_momentum, out=self.mu_var_momentum, 
                        dtype=self.model.dtype)
            np.add(self.running_var, self.mu_var_momentum, out=self.running_var, 
                   order="C", dtype=self.model.dtype, casting="unsafe")

        if self.spatial:
            match self.model.tensor_format:
                case PYDTNN_TENSOR_FORMAT.NCHW:
                    y = y.reshape((-1, self.ci, self.hi, self.wi), copy=False)
                case PYDTNN_TENSOR_FORMAT.NHWC:
                    y = y.reshape((-1, self.hi, self.wi, self.ci), copy=False)
                case _ :
                    raise ValueError(f"{self.model.tensor_format} tensor format not supported. Tensor format supported: {list(self.model.tensor_format)}")

        return np.asarray(y, dtype=self.model.dtype, order='C', copy=None)
    # --- END forward --- #

    def backward(self, dy: np.ndarray) -> np.ndarray:

        n = dy.shape[0]
        if self.spatial:
            dy = dy.reshape((-1, self.ci), copy=True)
            dx: np.ndarray = self.dx[: (n * self.hi * self.wi), :]
            dy_xn: np.ndarray = self.dy_xn[: (n * self.hi * self.wi), :]
        else:
            dx: np.ndarray = self.dx[:n, :]
            dy_xn: np.ndarray = self.dy_xn[:n, :]

        # dx = (self.gamma / (self.std * n)) * (n * dy - self.xn * self.dgamma - self.dbeta)
        np.multiply(dy, self.xn, out=dy_xn, dtype=self.model.dtype)
        np.sum(dy_xn, axis=0, out=self.dgamma, dtype=self.model.dtype)
        np.sum(dy, axis=0, out=self.dbeta, dtype=self.model.dtype)

        bn_training_bwd_cython(dx, dy, self.xn, self.std, self.gamma, self.dgamma, self.dbeta)

        if self.spatial:
            match self.model.tensor_format:
                case PYDTNN_TENSOR_FORMAT.NCHW:
                    dx = dx.reshape((-1, self.ci, self.hi, self.wi), copy=False)
                case PYDTNN_TENSOR_FORMAT.NHWC:
                    dx = dx.reshape((-1, self.hi, self.wi, self.ci), copy=False)
                case _ :
                    raise ValueError(f"{self.model.tensor_format} tensor format not supported. Tensor format supported: {list(self.model.tensor_format)}")

            
        return np.asarray(dx, dtype=self.model.dtype, order='C', copy=None)
    # --- END backward --- #
