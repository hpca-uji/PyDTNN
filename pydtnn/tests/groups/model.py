"""Model test group"""

from pydtnn.tests.model_tensor import ModelTensorTestCase
from pydtnn.tests.model_dtype import ModelDTypeTestCase
from warnings import warn
import logging
logger = logging.getLogger(__name__)


try:
    from pydtnn.tests.model_gpu import ModelGpuTestCase
except Exception:
    logger.warning("GPU not available, skiping tests!")
    warn("GPU not available, skiping tests!")
