# """
# PyDTNN CPU Activations

# If you want to add a new GPU activation layer:
#     1) create a new Python file in this directory,
#     2) define your CPU activation layer class as derived from ActivationCPU and, optionally, other Activation
#        derived class,
#     3) and, optionally, import your CPU activation layer on this file.
# """

# from pydtnn.utils import get_derived_classes
# from pydtnn.backends.cpu.activations.activation_cpu import ActivationCPU
# from pydtnn.backends.cpu.activations.arctanh_cpu import ArctanhCPU
# from pydtnn.backends.cpu.activations.log_cpu import LogCPU
# from pydtnn.backends.cpu.activations.relu_cpu import ReluCPU
# from pydtnn.backends.cpu.activations.sigmoid_cpu import SigmoidCPU
# from pydtnn.backends.cpu.activations.softmax_cpu import SoftmaxCPU
# from pydtnn.backends.cpu.activations.tanh_cpu import TanhCPU

# # Search this module for ActivationCPU derived classes and expose them
# get_derived_classes(ActivationCPU, locals())
