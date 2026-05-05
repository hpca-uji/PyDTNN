import logging
from warnings import warn


__all__ = (
    "LayerPyTorchTestCase",
    "BatchNormalizationReluTestCase",
    "Conv2DReluTestCase",
    "Conv2DBatchNormalizationTestCase",
    "Conv2DBatchNormalizationReluTestCase",
    "Conv2DGroupTestCase",
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
    warn("PyTorch not available, skiping tests!", ImportWarning)

# Fused
from pydtnn.tests.batch_normalization_relu import BatchNormalizationReluTestCase  # isort:skip  # noqa: E402
from pydtnn.tests.conv_2d_relu import Conv2DReluTestCase  # isort:skip  # noqa: E402
from pydtnn.tests.conv_2d_batch_normalization import Conv2DBatchNormalizationTestCase  # isort:skip  # noqa: E402
from pydtnn.tests.conv_2d_batch_normalization_relu import Conv2DBatchNormalizationReluTestCase  # isort:skip  # noqa: E402
from pydtnn.tests.conv_2d_group import Conv2DGroupTestCase  # isort:skip  # noqa: E402

# Models
from pydtnn.tests.model_dtype import ModelDTypeTestCase  # isort:skip  # noqa: E402
from pydtnn.tests.model_tensor import ModelTensorTestCase  # isort:skip  # noqa: E402

try:
    from pydtnn.tests.model_gpu import ModelGpuTestCase  # isort:skip  # noqa: E402
except Exception:
    warn("GPU not available, skiping tests!")

# Libraries
try:
    from pydtnn.tests.conv_2d_conv_gemm import Conv2DConvGemmTestCase
    from pydtnn.tests.conv_gemm import ConvGemmTestCase
    from pydtnn.tests.model_conv_gemm import ModelConvGemmTestCase
except Exception:
    warn("ConvGemm not available, skiping tests!")

try:
    from pydtnn.tests.conv_winograd import ConvWinogradTestCase
except Exception:
    warn("ConvWinograd not available, skiping tests!")

try:
    from pydtnn.tests.conv_direct import ConvDirectTestCase
except Exception:
    warn("ConvDirect not available, skiping tests!")
