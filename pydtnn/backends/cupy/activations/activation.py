from pydtnn.backends.numpy.activations.activation import ActivationNumpy
from pydtnn.backends.cupy.abstract.layerable import LayerableCupy
import logging
logger = logging.getLogger(__name__)


class ActivationCupy(ActivationNumpy, LayerableCupy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cuda_compiler = "nvcc"
