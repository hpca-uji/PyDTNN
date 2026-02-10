from pydtnn.layers.conv_2d import Conv2D
from pydtnn.backends.numpy.layers.layer import LayerNumpy
from pydtnn.utils.performance_models import im2col_time, matmul_time
from pydtnn.utils.constants import ArrayShape

from pydtnn.libs import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np


class AbstractConv2DNumpy(Conv2D[np.ndarray], LayerNumpy):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # More parameters initialized in initialize()
        self.biases = None  # type: ignore
        self.weights = None  # type: ignore
        self.fwd_time = None  # type: ignore
        self.bwd_time = None  # type: ignore
    # ----

    def _model_init(self, prev_shape: ArrayShape, x: np.ndarray | None = None) -> None:
        super()._model_init(prev_shape, x)
        if self.use_bias:
            bias_shape = (self.co,)  # NOTE: Is the same shape in every variant and grouping
            self.biases = np.asarray(self.biases_initializer(bias_shape, self.model.dtype), order="C")
            self.memory_used += self.biases.nbytes

        self.weights = np.asarray(self.weights_initializer(self.weights_shape, self.model.dtype), order="C")

        self.memory_used += self.weights.nbytes

        if not self.model.evaluate_only:
            if self.use_bias:
                self.db = np.zeros(shape=bias_shape, dtype=self.model.dtype, order="C")
                self.memory_used += self.db.nbytes

            self.dw: np.ndarray = np.zeros(self.weights.shape, dtype=self.model.dtype)
            self.memory_used += self.dw.nbytes

        # Performance models
        self.fwd_time = \
            im2col_time(m=(self.ci * self.kh * self.kw), n=(self.model.batch_size * self.ho * self.wo),
                        cpu_speed=self.model.cpu_speed, memory_bw=self.model.memory_bw,
                        dtype=self.model.dtype) + \
            matmul_time(m=self.co, n=(self.model.batch_size * self.ho * self.wo), k=(self.ci * self.kh * self.kw),
                        cpu_speed=self.model.cpu_speed, memory_bw=self.model.memory_bw,
                        dtype=self.model.dtype)  # type: ignore (It works well.)
        self.bwd_time = \
            matmul_time(m=self.co, n=(self.ci * self.kh * self.kw), k=(self.model.batch_size * self.ho * self.wo),
                        cpu_speed=self.model.cpu_speed, memory_bw=self.model.memory_bw,
                        dtype=self.model.dtype)
        self.bwd_time += matmul_time(m=(self.ci * self.kh * self.kw), n=(self.model.batch_size * self.ho * self.wo),
                                     k=self.co, cpu_speed=self.model.cpu_speed,
                                     memory_bw=self.model.memory_bw, dtype=self.model.dtype)

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Use a real forward variant!")

    def backward(self, dy: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Use a real backwards variant!")

    def print_in_convdirect_format(self) -> None:
        if self.wstride != 1 or self.hstride != 1:
            return
        # #l kn wo ho t kh kw ci wi hi"
        ci, hi, wi = self.model.decode_shape(self.prev_shape)
        print(self.id, self.co, self.wo, self.ho, self.model.batch_size, self.kh, self.kw, ci, wi, hi, sep="\t")
