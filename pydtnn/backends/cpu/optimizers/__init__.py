"""
CPU optimizers

If you want to add a new CPU optimizer:
    1) create a new Python file in this directory,
    2) define your optimizer class as derived from Optimizer and, optionally, other Optimizer derived class,
    3) and, optionally, import your optimizer on this file.
"""

from pydtnn.backends.cpu.optimizers.optimizer_cpu import OptimizerCPU
from pydtnn.utils import get_derived_classes

# Search this module for Optimizer derived classes and expose them
get_derived_classes(OptimizerCPU, locals())
