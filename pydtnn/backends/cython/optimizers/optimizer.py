import logging
from pydtnn.backends.numpy.optimizers.optimizer import OptimizerNumpy

__all__ = ("OptimizerCython",)

logger = logging.getLogger(__name__)

class OptimizerCython(OptimizerNumpy):
    """
    Extends an Optimizer class with the attributes and methods required by CPU Optimizers.
    """
