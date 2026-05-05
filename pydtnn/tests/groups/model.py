"""Model test group"""

import logging
from warnings import warn

__all__ = ("ModelDTypeTestCase", "ModelTensorTestCase", "ModelGpuTestCase")

logger = logging.getLogger(__name__)

from pydtnn.tests.model_dtype import ModelDTypeTestCase  # isort:skip  # noqa: E402
from pydtnn.tests.model_tensor import ModelTensorTestCase  # isort:skip  # noqa: E402

try:
    from pydtnn.tests.model_gpu import ModelGpuTestCase
except Exception:
    logger.warning("GPU not available, skiping tests!")
    warn("GPU not available, skiping tests!")
