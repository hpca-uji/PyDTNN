"""
PyDTNN CPU Activations

If you want to add a new GPU activation layer:
    1) create a new Python file in this directory,
    2) define your CPU activation layer class as derived from ActivationCPU and, optionally, other Activation
       derived class,
    3) and, optionally, import your CPU activation layer on this file.
"""

from pydtnn.utils import get_derived_classes
from .activation_cpu import ActivationCPU
from .arctanh_cpu import ArctanhCPU
from .log_cpu import LogCPU
from .relu_cpu import ReluCPU
from .sigmoid_cpu import SigmoidCPU
from .softmax_cpu import SoftmaxCPU
from .tanh_cpu import TanhCPU

# Search this module for ActivationCPU derived classes and expose them
get_derived_classes(ActivationCPU, locals())
