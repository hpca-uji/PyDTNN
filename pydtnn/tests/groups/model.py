"""Model test group"""

import logging
from warnings import warn

from pydtnn.tests.model_dtype import ModelDTypeTestCase
from pydtnn.tests.model_tensor import ModelTensorTestCase

logger = logging.getLogger(__name__)


try:
    from pydtnn.tests.model_gpu import ModelGpuTestCase
except Exception:
    logger.warning("GPU not available, skiping tests!")
    warn("GPU not available, skiping tests!")
