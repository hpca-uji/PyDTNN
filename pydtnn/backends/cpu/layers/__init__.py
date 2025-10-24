# """
# PyDTNN CPU Layers

# If you want to add a new CPU layer:
#     1) create a new Python file in this directory,
#     2) define your layer class as derived from LayerCPU and, optionally, other Layer derived class,
#     3) and, optionally, import your CPU layer on this file.
# """

# from pydtnn.backends.cpu.layers.layer_cpu import LayerCPU
# from pydtnn.utils import get_derived_classes

# # Search this module for LayerGPU derived classes and expose them
# get_derived_classes(LayerCPU, locals())
