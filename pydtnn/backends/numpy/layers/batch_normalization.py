from pydtnn.libs import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np
#from pydtnn.backends.numpy.utils.bn_training_cython import bn_training_bwd_cython , bn_training_fwd_cython

from pydtnn.layers.batch_normalization import BatchNormalization
from pydtnn.model import Model
from pydtnn.backends.numpy.layers.layer import LayerNumpy
from pydtnn.utils.tensor import TensorFormat, format_transpose
from pydtnn.utils.constants import ArrayShape, Parameters


class BatchNormalizationNumpy(BatchNormalization[np.ndarray], LayerNumpy):

    @property
    def _ary_prop(self) -> set[str]:
        return {Parameters.RUNNING_MEAN,
                Parameters.RUNNING_VAR,
                *super()._ary_prop}

    @staticmethod
    def get_inv_std(running_var: np.ndarray, epsilon: float, dtype: np.dtype) -> np.ndarray:
        inv_std = np.add(running_var, epsilon, dtype=dtype)
        np.sqrt(inv_std, out=inv_std,
                dtype=dtype)
        np.reciprocal(inv_std, out=inv_std,
                      dtype=dtype)
        return inv_std

    def initialize(self, prev_shape: ArrayShape, x: np.ndarray | None = None):
        super().initialize(prev_shape, x)

        if self.spatial:
            self.ci, self.hi, self.wi = self.model.decode_shape(self.shape)
            vars_shape = (self.model.batch_size * self.hi * self.wi, self.ci)
        else:
            self.ci = self.shape[0]
            # NOTE: in this case, self.hi and self.wi are 0 (self.shape should be somethin like: "(512, )"
            vars_shape = (self.model.batch_size, self.ci)
        shape_ = (self.ci,)

        self.gamma = np.full(shape_, self.gamma_init_val, dtype=self.model.dtype)
        self.beta = np.full(shape_, self.beta_init_val, dtype=self.model.dtype)
        self.running_mean = np.asarray(self.moving_mean_initializer(shape_, self.model.dtype))
        self.running_var = np.asarray(self.moving_variance_initializer(shape_, self.model.dtype))

        self.nparams = self.gamma.size + self.beta.size + self.running_mean.size + self.running_var.size

        # NOTE: These attributes only store data, their value before the operation doesn't matter; they're initalized due avoid warnings in "LayerAndActivationBase.export".
        self.y_dx: np.ndarray = np.zeros(vars_shape, dtype=self.model.dtype)
        self.real_memory_size += (self.nparams * self.model.dtype.itemsize) + self.y_dx.nbytes
        # NOTE: This variable stores both y and dx values.

        self._mean_inv_shape = shape_
        self._var_inv_shape = shape_
        self.std_shape = shape_

        self.std: np.ndarray = np.zeros(shape=self.std_shape, dtype=self.model.dtype)
        self.real_memory_size += self.std.nbytes

        self.temp_memory_size += int(np.prod(self._mean_inv_shape) + np.prod(self._var_inv_shape)) * self.model.dtype.itemsize

        # self.dx: np.ndarray = np.zeros(shape=vars_shape, dtype=self.model.dtype)
        # self.real_memory_size += self.dx.nbytes
        self.dgamma: np.ndarray = np.zeros(shape=shape_, dtype=self.model.dtype)
        self.real_memory_size += self.dgamma.nbytes
        self.dbeta: np.ndarray = np.zeros(shape=shape_, dtype=self.model.dtype)
        self.real_memory_size += self.dbeta.nbytes

        self._mean_shape = (self.ci, )
        self._var_shape = (self.ci, )
        self.dy_xn_shape = vars_shape
        self.temp_memory_size += int(np.prod(self._mean_shape) + np.prod(self._var_shape) + np.prod(self.dy_xn_shape)) * self.model.dtype.itemsize

        self.real_memory_size += self.temp_memory_size
    # --

    def post_initialize(self) -> None:
        super().post_initialize()
        with self.model.memory:
            self._mean_inv = self.model.memory.ndarray(self._mean_inv_shape, dtype=self.model.dtype)
            self._var_inv = self.model.memory.ndarray(self._var_inv_shape, dtype=self.model.dtype)
            self._mean = self.model.memory.ndarray(self._mean_shape, dtype=self.model.dtype)
            self._var = self.model.memory.ndarray(self._var_shape, dtype=self.model.dtype)
            self.dy_xn = self.model.memory.ndarray(self.dy_xn_shape, dtype=self.model.dtype)

    def forward(self, x: np.ndarray) -> np.ndarray:

        self.y_dx: np.ndarray
        n = x.shape[0]

        if self.spatial:
            # NOTE: Executing in this format gives better results.
            x = format_transpose(x, self.model.tensor_format, TensorFormat.NHWC)
            x = x.reshape((-1, self.ci))
        # else: x = x (no reshape needed)

        y: np.ndarray = self.y_dx[:x.shape[0], :]
        self.xn = x

        if self.model.mode is Model.Mode.EVALUATE:
            _mean = self.running_mean
            _var = self.running_var
        else:  # Model.Mode.TRAIN:
            _mean = self._mean
            _var = self._var
            np.mean(self.xn, axis=0, dtype=self.model.dtype, out=_mean)
            np.var(self.xn, axis=0, dtype=self.model.dtype, out=_var)

            inv_momentum = (1.0 - self.momentum)
            # self.running_mean = self.momentum * self.running_mean + inv_momentum * _mean
            np.multiply(self.momentum, self.running_mean, out=self.running_mean,
                        dtype=self.model.dtype)
            np.multiply(inv_momentum, _mean, out=self._mean_inv,
                        dtype=self.model.dtype)
            np.add(self.running_mean, self._mean_inv, out=self.running_mean,
                   dtype=self.model.dtype)

            # self.running_var = self.momentum * self.running_var + inv_momentum * _var
            np.multiply(self.momentum, self.running_var, out=self.running_var,
                        dtype=self.model.dtype)
            np.multiply(inv_momentum, _var, out=self._var_inv,
                        dtype=self.model.dtype)
            np.add(self.running_var, self._var_inv, out=self.running_var,
                   dtype=self.model.dtype)
        # anyways:

        # bn_training_fwd_cython(x, y, self.xn, self.std, self.gamma, self.beta, _mean, _var, self.epsilon)
        # y = ((x - mean(x)) / sqrt(var(x) + epsilon)) * gamma + beta
        np.subtract(self.xn, _mean, out=self.xn,
                    dtype=self.model.dtype)

        np.add(_var, self.epsilon, out=self.std,
               dtype=self.model.dtype)
        np.sqrt(self.std, out=self.std,
                dtype=self.model.dtype)

        np.divide(self.xn, self.std, out=self.xn,
                  dtype=self.model.dtype)
        np.multiply(self.gamma, self.xn, out=y,
                    dtype=self.model.dtype)
        np.add(y, self.beta, out=y,
               dtype=self.model.dtype)

        if self.spatial:
            y = y.reshape((n, self.hi, self.wi, self.ci))
            y = format_transpose(y, TensorFormat.NHWC, self.model.tensor_format)

        return np.asarray(y, dtype=self.model.dtype, order="C")

    def backward(self, dy: np.ndarray) -> np.ndarray:

        n = dy.shape[0]
        if self.spatial:
            num_elems = (n * self.hi * self.wi)

            # NOTE: Executing in this format gives better results.
            dy = format_transpose(dy, self.model.tensor_format, TensorFormat.NHWC)
            dy = dy.reshape((num_elems, self.ci))
        else:
            num_elems = n

        dx: np.ndarray = self.y_dx[: num_elems, :]
        dy_xn: np.ndarray = self.dy_xn[: num_elems, :]
        dy_xn.fill(0)

        np.multiply(dy, self.xn, out=dy_xn, dtype=self.model.dtype)
        np.sum(dy_xn, axis=0, out=self.dgamma, dtype=self.model.dtype)
        np.sum(dy, axis=0, out=self.dbeta, dtype=self.model.dtype)

        # dx = (self.gamma / (self.std * n)) * (n * dy - self.xn * self.dgamma - self.dbeta)
        np.multiply(self.std, n, out=dx)
        np.divide(self.gamma, dx, out=dx)
        np.multiply(n, dy, out=dy)
        np.multiply(self.xn, self.dgamma, out=self.xn)
        np.subtract(dy, self.xn, out=dy)
        np.subtract(dy, self.dbeta, out=dy)
        np.multiply(dx, dy, out=dx)

        if self.spatial:
            dx = dx.reshape((n, self.hi, self.wi, self.ci), copy=False)
            dx = format_transpose(dx, TensorFormat.NHWC, self.model.tensor_format)
        # else: nothing special (It has the right format)

        return np.asarray(dx, dtype=self.model.dtype, order="C")
