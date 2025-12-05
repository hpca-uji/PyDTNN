"""Library test group"""
from warnings import warn

# ConvGemm
try:
    from pydtnn.tests.conv_gemm import ConvGemmTestCase
    from pydtnn.tests.conv_2d_conv_gemm import Conv2DConvGemmTestCase
    from pydtnn.tests.model_conv_gemm import ModelConvGemmTestCase
except Exception:
    warn("ConvGemm not available, skiping tests!")

# ConvWinograd
try:
    from pydtnn.tests.conv_winograd import ConvWinogradTestCase
except Exception:
    warn("ConvWinograd not available, skiping tests!")

# ConvDirect
try:
    from pydtnn.tests.conv_direct import ConvDirectTestCase
except Exception:
    warn("ConvDirect not available, skiping tests!")
