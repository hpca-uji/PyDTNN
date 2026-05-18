"""
Cython-based optimizer backend for the PyDTNN framework.
"""

import logging

from pydtnn.backends.numpy.optimizers.abstract.optimizer import OptimizerNumpy

__all__ = ("OptimizerCython",)

logger = logging.getLogger(__name__)


class OptimizerCython(OptimizerNumpy):
    """
    Extends an Optimizer class with the attributes and methods required by CPU Optimizers.
    """
