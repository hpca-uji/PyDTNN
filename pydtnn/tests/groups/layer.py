"""Layer test group"""

import logging
from warnings import warn

logger = logging.getLogger(__name__)


try:
    from pydtnn.tests.layer_pytorch import LayerPyTorchTestCase
except Exception:
    logger.warning("PyTorch not available, skiping tests!")
    warn("PyTorch not available, skiping tests!", ImportWarning)
