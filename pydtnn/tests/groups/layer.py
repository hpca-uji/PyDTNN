"""Layer test group"""

import logging
logger = logging.getLogger(__name__)

from warnings import warn

try:
    from pydtnn.tests.layer_pytorch import LayerPyTorchTestCase
except Exception:
    logger.warning("PyTorch not available, skiping tests!")
    warn("PyTorch not available, skiping tests!", ImportWarning)
