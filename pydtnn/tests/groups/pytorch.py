"""PyTorch test group"""

import logging

__all__ = ("PytorchLayerTestCase", "PytorchModelTestCase")

logger = logging.getLogger(__name__)


try:
    from pydtnn.tests.pytorch_layer import PytorchLayerTestCase
    from pydtnn.tests.pytorch_model import PytorchModelTestCase
except Exception:
    logger.warning("PyTorch not available, skiping tests!")
