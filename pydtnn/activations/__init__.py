# """
# PyDTNN Activation layers

# If you want to add a new activation layer:
#     1) create a new Python file in this directory,
#     2) define your activation layer class as derived from Activation (or any Activation derived class),
#     3) and, optionally, import your activation layer on this file.
# """

# from pydtnn.activations.activation import Activation
# from pydtnn.activations.arctanh import Arctanh
# from pydtnn.activations.log import Log
# from pydtnn.activations.relu import Relu
# from pydtnn.activations.relu6 import Relu6
# from pydtnn.activations.leaky_relu import LeakyRelu
# from pydtnn.activations.sigmoid import Sigmoid
# from pydtnn.activations.softmax import Softmax
# from pydtnn.activations.tanh import Tanh
# from pydtnn.utils import get_derived_classes

# # Aliases
# sigmoid = Sigmoid
# relu = Relu
# relu6 = Relu6
# leaky_relu = leakyrelu = LeakyRelu
# tanh = Tanh
# arctanh = Arctanh
# log_sigmoid = LogSigmoid = log = Log
# softmax = Softmax

# # Search this module for Activation derived classes and expose them
# get_derived_classes(Activation, locals())
