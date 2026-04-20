import cupy as np
from pydtnn.utils.constants import ArrayShape
from pydtnn.backends.numpy.layers.fc import FCNumpy
from pydtnn.backends.cupy.layers.layer import LayerCupy
import logging
logger = logging.getLogger(__name__)


class FCCupy(FCNumpy, LayerCupy):

    def _model_init(self, prev_shape: ArrayShape, x: np.ndarray | None = None) -> None:
        super()._model_init(prev_shape, x)

        self.stream_2 = np.cuda.Stream()
