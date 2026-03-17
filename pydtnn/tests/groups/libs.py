"""Library test group"""
import logging
logger = logging.getLogger(__name__)

from warnings import warn

# ConvGemm
try:
    from pydtnn.tests.conv_gemm import ConvGemmTestCase
    from pydtnn.tests.conv_2d_conv_gemm import Conv2DConvGemmTestCase
    from pydtnn.tests.model_conv_gemm import ModelConvGemmTestCase
except Exception:
    logger.warning("ConvGemm not available, skiping tests!")
    warn("ConvGemm not available, skiping tests!", ImportWarning)

# ConvWinograd
try:
    from pydtnn.tests.conv_winograd import ConvWinogradTestCase
except Exception:
    logger.warning("ConvWinograd not available, skiping tests!")
    warn("ConvWinograd not available, skiping tests!", ImportWarning)

# ConvDirect
try:
    from pydtnn.tests.conv_direct import ConvDirectTestCase
except Exception:
    logger.warning("ConvDirect not available, skiping tests!")
    warn("ConvDirect not available, skiping tests!", ImportWarning)
