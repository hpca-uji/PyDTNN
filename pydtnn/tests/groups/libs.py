"""Library test group"""

import logging

__all__ = ("Conv2DConvGemmTestCase", "ConvGemmTestCase", "ModelConvGemmTestCase", "ConvWinogradTestCase", "ConvDirectTestCase")

logger = logging.getLogger(__name__)


# ConvGemm
try:
    from pydtnn.tests.conv_2d_conv_gemm import Conv2DConvGemmTestCase
    from pydtnn.tests.conv_gemm import ConvGemmTestCase
    from pydtnn.tests.model_conv_gemm import ModelConvGemmTestCase
except Exception:
    logger.warning("ConvGemm not available, skiping tests!")

# ConvWinograd
try:
    from pydtnn.tests.conv_winograd import ConvWinogradTestCase
except Exception:
    logger.warning("ConvWinograd not available, skiping tests!")

# ConvDirect
try:
    from pydtnn.tests.conv_direct import ConvDirectTestCase
except Exception:
    logger.warning("ConvDirect not available, skiping tests!")
