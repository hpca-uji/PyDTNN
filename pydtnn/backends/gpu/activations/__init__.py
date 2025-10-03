"""
PyDTNN GPU Activations

If you want to add a new GPU activation layer:
    1) create a new Python file in this directory,
    2) define your GPU activation layer class as derived from ActivationGPU and, optionally, other Activation
       derived class,
    3) and, optionally, import your GPU activation layer on this file.
"""

from .activation_gpu import ActivationGPU
from .arctanh_gpu import ArctanhGPU
from .log_gpu import LogGPU
from .relu_gpu import ReluGPU
from .sigmoid_gpu import SigmoidGPU
from .softmax_gpu import SoftmaxGPU
from .tanh_gpu import TanhGPU
from pydtnn.utils import get_derived_classes

# Search this module for ActivationGPU derived classes and expose them
get_derived_classes(ActivationGPU, locals())
