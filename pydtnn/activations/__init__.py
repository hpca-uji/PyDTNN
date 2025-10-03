"""
PyDTNN Activation layers

If you want to add a new activation layer:
    1) create a new Python file in this directory,
    2) define your activation layer class as derived from Activation (or any Activation derived class),
    3) and, optionally, import your activation layer on this file.
"""

from .activation import Activation
from .arctanh import Arctanh
from .log import Log
from .relu import Relu
from .relu6 import Relu6
from .leaky_relu import LeakyRelu
from .sigmoid import Sigmoid
from .softmax import Softmax
from .tanh import Tanh
from ..utils import get_derived_classes

# Aliases
sigmoid = Sigmoid
relu = Relu
relu6 = Relu6
leaky_relu = leakyrelu = LeakyRelu
tanh = Tanh
arctanh = Arctanh
log = Log
softmax = Softmax

# Search this module for Activation derived classes and expose them
get_derived_classes(Activation, locals())
