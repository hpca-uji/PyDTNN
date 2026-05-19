"""
Collection of test cases for the PyDTNN framework.

This module aggregates various test suites for layers, fused operations,
models, and specific library implementations to facilitate centralized
test execution.
"""

import logging

__all__ = (
    "LayerPyTorchTestCase",
    "BatchNormalizationReluTestCase",
    "Conv2DReluTestCase",
    "Conv2DBatchNormalizationTestCase",
    "Conv2DBatchNormalizationReluTestCase",
    "ModelDTypeTestCase",
    "ModelTensorTestCase",
    "ModelGpuTestCase",
    "Conv2DConvGemmTestCase",
    "ConvGemmTestCase",
    "ModelConvGemmTestCase",
    "ConvWinogradTestCase",
    "ConvDirectTestCase",
)


logger = logging.getLogger(__name__)


# Layers
try:
    from pydtnn.tests.layer_pytorch import LayerPyTorchTestCase
except Exception:
    logger.warning("PyTorch not available, skiping tests!")

# Fused
from pydtnn.tests.batch_normalization_relu import BatchNormalizationReluTestCase  # isort:skip  # noqa: E402
from pydtnn.tests.conv_2d_relu import Conv2DReluTestCase  # isort:skip  # noqa: E402
from pydtnn.tests.conv_2d_batch_normalization import Conv2DBatchNormalizationTestCase  # isort:skip  # noqa: E402
from pydtnn.tests.conv_2d_batch_normalization_relu import Conv2DBatchNormalizationReluTestCase  # isort:skip  # noqa: E402

# Models
from pydtnn.tests.model_dtype import ModelDTypeTestCase  # isort:skip  # noqa: E402
from pydtnn.tests.model_tensor import ModelTensorTestCase  # isort:skip  # noqa: E402

try:
    from pydtnn.tests.model_gpu import ModelGpuTestCase  # isort:skip  # noqa: E402
except Exception:
    logger.warning("GPU not available, skiping tests!")

# Libraries
try:
    from pydtnn.tests.conv_2d_conv_gemm import Conv2DConvGemmTestCase
    from pydtnn.tests.conv_gemm import ConvGemmTestCase
    from pydtnn.tests.model_conv_gemm import ModelConvGemmTestCase
except Exception:
    logger.warning("ConvGemm not available, skiping tests!")

try:
    from pydtnn.tests.conv_winograd import ConvWinogradTestCase
except Exception:
    logger.warning("ConvWinograd not available, skiping tests!")

try:
    from pydtnn.tests.conv_direct import ConvDirectTestCase
except Exception:
    logger.warning("ConvDirect not available, skiping tests!")
