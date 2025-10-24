# """
# PyDTNN GPU Activations

# If you want to add a new GPU activation layer:
#     1) create a new Python file in this directory,
#     2) define your GPU activation layer class as derived from ActivationGPU and, optionally, other Activation
#        derived class,
#     3) and, optionally, import your GPU activation layer on this file.
# """

# from pydtnn.backends.gpu.activations.activation_gpu import ActivationGPU
# from pydtnn.backends.gpu.activations.arctanh_gpu import ArctanhGPU
# from pydtnn.backends.gpu.activations.log_gpu import LogGPU
# from pydtnn.backends.gpu.activations.relu_gpu import ReluGPU
# from pydtnn.backends.gpu.activations.sigmoid_gpu import SigmoidGPU
# from pydtnn.backends.gpu.activations.softmax_gpu import SoftmaxGPU
# from pydtnn.backends.gpu.activations.tanh_gpu import TanhGPU
# from pydtnn.utils import get_derived_classes

# # Search this module for ActivationGPU derived classes and expose them
# get_derived_classes(ActivationGPU, locals())
