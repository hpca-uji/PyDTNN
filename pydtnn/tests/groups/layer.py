"""Layer test group"""

import logging

__all__ = ("LayerPyTorchTestCase",)

logger = logging.getLogger(__name__)


try:
    from pydtnn.tests.pytorch_layer import LayerPyTorchTestCase
except Exception:
    logger.warning("PyTorch not available, skiping tests!")
