from warnings import warn

# Implementation
try:
    from pydtnn.tests.layer_pytorch import LayerPyTorchTestCase
except Exception:
    warn("PyTorch not available, skiping tests!")

from pydtnn.tests.tensor_format import TensorFormatTestCase

# Convolutions
from pydtnn.tests.conv2d_conv_group import Conv2DConvGroupTestCase
from pydtnn.tests.conv2d_relu import Conv2DReluTestCase
from pydtnn.tests.conv2d_batch_normalization_relu import Conv2DBatchNormalizationReluTestCase

# Libraries
try:
    from pydtnn.tests.conv_gemm import ConvGemmTestCase
    from pydtnn.tests.model_conv_gemm import ModelConvGemmTestCase
    from pydtnn.tests.conv2d_conv_gemm import Conv2DConvGemmTestCase
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

# Models
try:
    from pydtnn.tests.model_gpu import ModelGpuTestCase
except Exception:
    warn("GPU not available, skiping tests!")
