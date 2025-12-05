from warnings import warn

# Layers
try:
    from pydtnn.tests.layer_pytorch import LayerPyTorchTestCase
except Exception:
    warn("PyTorch not available, skiping tests!")

# Fused
from pydtnn.tests.conv_2d_group import Conv2DGroupTestCase
from pydtnn.tests.conv_2d_relu import Conv2DReluTestCase
from pydtnn.tests.conv_2d_batch_normalization import Conv2DBatchNormalizationTestCase
from pydtnn.tests.conv_2d_batch_normalization_relu import Conv2DBatchNormalizationReluTestCase
from pydtnn.tests.batch_normalization_relu import BatchNormalizationReluTestCase

# Models
from pydtnn.tests.model_dtype import ModelDTypeTestCase
from pydtnn.tests.model_tensor import ModelTensorTestCase
try:
    from pydtnn.tests.model_gpu import ModelGpuTestCase
except Exception:
    warn("GPU not available, skiping tests!")

# Libraries
try:
    from pydtnn.tests.conv_gemm import ConvGemmTestCase
    from pydtnn.tests.conv_2d_conv_gemm import Conv2DConvGemmTestCase
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
