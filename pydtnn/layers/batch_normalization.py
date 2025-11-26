import numpy as np

from pydtnn.layers.layer import Layer

from typing import Any, Callable

from pydtnn.utils.initializers import zeros, ones

from pydtnn.utils.constants import Array, ArrayShape, Parameters

class BatchNormalization[T: Array](Layer[T]):

    def __init__(self, beta=0.0, gamma=1.0, momentum=0.9, epsilon=1e-5,
                 moving_mean_initializer: Callable = zeros,
                 moving_variance_initializer: Callable = ones,
                 sync_stats=False):
        super().__init__()
        self.gamma_init_val = gamma
        self.beta_init_val = beta
        self.momentum = momentum
        self.epsilon = epsilon
        self.moving_mean_initializer: Callable[[ArrayShape, np.dtype], np.ndarray] = moving_mean_initializer
        self.moving_variance_initializer: Callable[[ArrayShape, np.dtype], np.ndarray] = moving_variance_initializer
        self.grad_vars = {Parameters.BETA: Parameters.DBETA, Parameters.GAMMA: Parameters.DGAMMA}
        self.sync_stats = sync_stats
        # The following attributes will be initialized later
        self.co = self.ci = self.hi = self.wi = 0
        self.spatial: bool = None  # type: ignore
        self.gamma: T = None  # type: ignore
        self.beta: T = None  # type: ignore
        self.running_mean: T = None  # type: ignore
        self.running_var: T = None  # type: ignore
        self.std: np.ndarray = None  # type: ignore
        self.xn: np.ndarray = None  # type: ignore
        self.dgamma: T = None  # type: ignore
        self.dbeta: T = None  # type: ignore

    def export(self) -> dict[str, Any]:
        data = super().export()

        data[Parameters.RUNNING_MEAN] = self._export_prop(Parameters.RUNNING_MEAN)
        data[Parameters.RUNNING_VAR] = self._export_prop(Parameters.RUNNING_VAR)

        return data
    # ---

    def import_(self, data: dict[str, Any]) -> None:

        self._import_prop(Parameters.RUNNING_MEAN, data[Parameters.RUNNING_MEAN])
        self._import_prop(Parameters.RUNNING_VAR, data[Parameters.RUNNING_VAR])

        return super().import_(data)

    def initialize(self, prev_shape: ArrayShape, x: T | None):
        super().initialize(prev_shape, x)
        self.shape = prev_shape
        self.spatial = len(self.shape) > 2
