"""Model test group"""

from warnings import warn

from pydtnn.tests.model_dtype import ModelDTypeTestCase
from pydtnn.tests.model_tensor import ModelTensorTestCase
try:
    from pydtnn.tests.model_gpu import ModelGpuTestCase
except Exception:
    warn("GPU not available, skiping tests!")
