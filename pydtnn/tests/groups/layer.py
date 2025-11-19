"""Layer test group"""

from warnings import warn

# Implementation
try:
    from pydtnn.tests.layer_pytorch import LayerPyTorchTestCase
except Exception:
    warn("PyTorch not available, skiping tests!")
