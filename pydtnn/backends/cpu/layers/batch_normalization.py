import numpy as np
from pydtnn.backends.cpu.utils.bn_training_cython import bn_training_bwd_cython  # , bn_training_fwd_cython

from pydtnn.layers.batch_normalization import BatchNormalization
from pydtnn.model import Model
from pydtnn.backends.cpu.layers.layer import LayerCPU
from pydtnn.utils.tensor import TensorFormat, format_transpose
from pydtnn.utils.constants import ArrayShape, Parameters


class BatchNormalizationCPU(BatchNormalization[np.ndarray], LayerCPU):

    @property
    def _ary_prop(self) -> set[str]:
        return {Parameters.RUNNING_MEAN,
                Parameters.RUNNING_VAR,
                *super()._ary_prop}

    @staticmethod
    def get_inv_std(running_var: np.ndarray, epsilon: float, dtype: np.dtype) -> np.ndarray:
        inv_std = np.add(running_var, epsilon, dtype=dtype, order="C")
        np.sqrt(inv_std, out=inv_std,
                dtype=dtype)
        np.reciprocal(inv_std, out=inv_std,
                      dtype=dtype)
        return inv_std

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
        self.dgamma: np.ndarray = np.zeros(shape=(self.ci,), dtype=self.model.dtype, order="C")
        self.dbeta: np.ndarray = np.zeros(shape=(self.ci,), dtype=self.model.dtype, order="C")
        self.running_mean = self.moving_mean_initializer(shape_, self.model.dtype)
        self.running_var = self.moving_variance_initializer(shape_, self.model.dtype)
        self.nparams = self.gamma.size + self.beta.size + self.running_mean.size + self.running_var.size

        # NOTE: These attributes only store data, their value before the operation doesn't matter; they're initalized due avoid warnings in "LayerAndActivationBase.export".
        self._mean_inv: np.ndarray = np.zeros(shape=(self.ci,), dtype=self.model.dtype, order="C")
        self._var_inv: np.ndarray = np.zeros(shape=(self.ci,), dtype=self.model.dtype, order="C")
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
        self.xn = x

        if self.model.mode is Model.Mode.EVALUATE:
            _mean = self.running_mean
            _var = self.running_var
        else:  # ModelModeEnum.TRAIN:
            _mean = np.mean(self.xn, axis=0, dtype=self.model.dtype)
            _var = np.var(self.xn, axis=0, dtype=self.model.dtype)

            inv_momentum = (1.0 - self.momentum)
            # self.running_mean = self.momentum * self.running_mean + inv_momentum * _mean
            np.multiply(self.momentum, self.running_mean, out=self.running_mean,
                        dtype=self.model.dtype)
            np.multiply(inv_momentum, _mean, out=self._mean_inv,
                        dtype=self.model.dtype, order="C")
            np.add(self.running_mean, self._mean_inv, out=self.running_mean,
                   dtype=self.model.dtype)

            # self.running_var = self.momentum * self.running_var + inv_momentum * _var
            np.multiply(self.momentum, self.running_var, out=self.running_var,
                        dtype=self.model.dtype)
            np.multiply(inv_momentum, _var, out=self._var_inv,
                        dtype=self.model.dtype, order="C")
            np.add(self.running_var, self._var_inv, out=self.running_var,
                   dtype=self.model.dtype)
        # anyways:

        # bn_training_fwd_cython(x, y, self.xn, self.std, self.gamma, self.beta, _mean, _var, self.epsilon)
        # y = ((x - mean(x)) / sqrt(var(x) + epsilon)) * gamma + beta
        np.subtract(self.xn, _mean, out=self.xn,
                    dtype=self.model.dtype)

        np.add(_var, self.epsilon, out=self.std,
               dtype=self.model.dtype, order="C")
        np.sqrt(self.std, out=self.std,
                dtype=self.model.dtype)

        np.divide(self.xn, self.std, out=self.xn,
                  dtype=self.model.dtype)
        np.multiply(self.gamma, self.xn, out=y,
                    dtype=self.model.dtype)
        np.add(y, self.beta, out=y,
               dtype=self.model.dtype)

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
