# """
# Loss GPU classes

# If you want to add a new GPU loss:
#     1) create a new Python file in this directory,
#     2) define your GPU loss class as derived from LossGPU and, optionally, other Loss derived class,
#     3) and, optionally, import your loss on this file.
# """

# from pydtnn.utils import get_derived_classes
# from pydtnn.backends.gpu.losses.binary_cross_entropy_gpu import BinaryCrossEntropyGPU
# from pydtnn.backends.gpu.losses.categorical_cross_entropy_gpu import CategoricalCrossEntropyGPU
# from pydtnn.backends.gpu.losses.loss_gpu import LossGPU

# # Search this module for LossGPU derived classes and expose them
# get_derived_classes(LossGPU, locals())
