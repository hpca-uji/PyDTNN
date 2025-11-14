import numpy as np
from pydtnn.cython.bn_training_cython import bn_training_bwd_cython

from pydtnn.layers.batch_normalization import BatchNormalization
from pydtnn.model import Model
from pydtnn.backends.cpu.layers.layer_cpu import LayerCPU
from pydtnn.utils.tensor import TensorFormat, format_transpose
from pydtnn.utils.types import ArrayShape

class BatchNormalizationCPU(LayerCPU, BatchNormalization[np.ndarray]):

    def initialize(self, prev_shape: ArrayShape, x: np.ndarray | None = None):
        super().initialize(prev_shape, x)

        if self.spatial:
            self.ci, self.hi, self.wi = self.model.decode_shape(self.shape)
            shape_ = (self.ci,)
        else:
            self.ci = self.shape[0]
            shape_ = (self.ci,)

        self.gamma = np.full(shape_, self.gamma_init_val, dtype=self.model.dtype, order="C")
        self.beta = np.full(shape_, self.beta_init_val, dtype=self.model.dtype, order="C")
        self.running_mean = self.moving_mean_initializer(shape_, self.model.dtype)
        self.running_var = self.moving_variance_initializer(shape_, self.model.dtype)
        # self.inv_std = 1.0 / np.sqrt(self.running_var + self.epsilon)
        self.inv_std = np.sqrt(self.running_var + self.epsilon, dtype=self.model.dtype, order="C")
        np.reciprocal(self.inv_std, out=self.inv_std, dtype=self.model.dtype)
        self.nparams = self.gamma.size + self.beta.size + self.running_mean.size + self.running_var.size

        # NOTE: These attributes only store data, their value before the operation doesn't matter; they're initalized due avoid warnings in "LayerAndActivationBase.export".
        self.mu: np.ndarray = np.zeros(shape=(self.ci,), dtype=self.model.dtype, order="C")
        self.mu_var_momentum: np.ndarray = np.zeros(shape=(self.ci,), dtype=self.model.dtype, order="C")
        self.var: np.ndarray = np.zeros(shape=(self.ci,), dtype=self.model.dtype, order="C")
        self.var_eps: np.ndarray = np.zeros(shape=(self.ci,), dtype=self.model.dtype, order="C")
        self.dgamma: np.ndarray = np.zeros(shape=(self.ci,), dtype=self.model.dtype, order="C")
        self.dbeta: np.ndarray = np.zeros(shape=(self.ci,), dtype=self.model.dtype, order="C")
        self.std: np.ndarray = np.zeros(shape=(self.ci,), dtype=self.model.dtype, order="C")
        if self.spatial:
            self.dx: np.ndarray = np.zeros(shape=(self.model.batch_size * self.hi * self.wi, self.ci), dtype=self.model.dtype, order="C")
            self.y = np.zeros((self.model.batch_size * self.hi * self.wi, self.ci), dtype=self.model.dtype, order="C")
            self.dy_xn = np.zeros((self.model.batch_size * self.hi * self.wi, self.ci), dtype=self.model.dtype, order="C")
        else:
            # NOTE: in this case, self.hi and self.wi are 0 (self.shape should be somethin like: "(512, )"
            self.dx: np.ndarray = np.zeros(shape=(self.model.batch_size, self.ci), dtype=self.model.dtype, order="C")
            self.y = np.zeros((self.model.batch_size, self.ci), dtype=self.model.dtype, order="C")
            self.dy_xn = np.zeros((self.model.batch_size, self.ci), dtype=self.model.dtype, order="C")
    # --

    def forward(self, x: np.ndarray) -> np.ndarray:

        self.y: np.ndarray
        n = x.shape[0]

        if self.spatial:
            # NOTE: Executing in this format gives better results.
            x = format_transpose(x, self.model.tensor_format, TensorFormat.NHWC)
            x = x.reshape((-1, self.ci), copy=None, order="C")
        # else: x = x (no reshape needed)

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

            self.xn = x

            # y = ((x - mean(x)) / sqrt(var(x) + epsilon)) * gamma + beta
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

            # NOTE: casting="unsafe", means that numpy will cast the data to the new type always.
            np.add(self.running_mean, self.mu_var_momentum, out=self.running_mean,
                   order="C", dtype=self.model.dtype, casting="unsafe")

            # self.running_var = self.momentum * self.running_var + (1.0 - self.momentum) * self.var
            np.multiply(self.running_var, self.momentum, out=self.running_var,
                        dtype=self.model.dtype)
            np.multiply(self.var, inv_momentum, out=self.mu_var_momentum,
                        dtype=self.model.dtype)
            # NOTE: casting="unsafe", means that numpy will cast the data to the new type always.
            np.add(self.running_var, self.mu_var_momentum, out=self.running_var,
                   order="C", dtype=self.model.dtype, casting="unsafe")

        if self.spatial:
            y = y.reshape((n, self.hi, self.wi, self.ci), copy=False)
            y = format_transpose(y, TensorFormat.NHWC, self.model.tensor_format)

        return np.asarray(y, dtype=self.model.dtype, order='C', copy=None)

    def backward(self, dy: np.ndarray) -> np.ndarray:

        n = dy.shape[0]
        if self.spatial:
            num_elems = (n * self.hi * self.wi)

            # NOTE: Executing in this format gives better results.
            dy = format_transpose(dy, self.model.tensor_format, TensorFormat.NHWC)
            dy = dy.reshape((num_elems, self.ci), copy=None)
        else:
            num_elems = n

        dx: np.ndarray = self.dx[: num_elems, :]
        dy_xn: np.ndarray = self.dy_xn[: num_elems, :]

        # dx = (self.gamma / (self.std * n)) * (n * dy - self.xn * self.dgamma - self.dbeta)
        np.multiply(dy, self.xn, out=dy_xn, dtype=self.model.dtype)
        np.sum(dy_xn, axis=0, out=self.dgamma, dtype=self.model.dtype)
        np.sum(dy, axis=0, out=self.dbeta, dtype=self.model.dtype)

        bn_training_bwd_cython(dx, dy, self.xn, self.std, self.gamma, self.dgamma, self.dbeta)

        if self.spatial:
            dx = dx.reshape((n, self.hi, self.wi, self.ci), copy=False)
            dx = format_transpose(dx, TensorFormat.NHWC, self.model.tensor_format)

        return np.asarray(dx, dtype=self.model.dtype, order='C', copy=None)
