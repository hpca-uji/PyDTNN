"""Model test group"""

import logging
logger = logging.getLogger(__name__)

from warnings import warn

from pydtnn.tests.model_dtype import ModelDTypeTestCase
from pydtnn.tests.model_tensor import ModelTensorTestCase
try:
    from pydtnn.tests.model_gpu import ModelGpuTestCase
except Exception:
    logger.warning("GPU not available, skiping tests!")
    warn("GPU not available, skiping tests!")
