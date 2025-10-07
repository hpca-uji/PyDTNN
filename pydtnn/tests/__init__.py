from pydtnn.tests.conv2d_conv_gemm import Conv2DConvGemmTestCase
# from pydtnn.tests.conv2d_conv_gemm_slow import Conv2DConvGemmSlowTestCase
from pydtnn.tests.conv_gemm import ConvGemmTestCase
from pydtnn.tests.conv_gemm_nhwc import ConvGemmNHWCTestCase
from pydtnn.tests.check_conv_gemm_models import CheckConvGemmModels
from pydtnn.tests.check_conv_gemm_nchw_models import CheckConvGemmNCHWModels
from pydtnn.tests.check_tensor_format_models import CheckTensorFormatModels
try:
    from pydtnn.tests.check_gpu_models import CheckGPUModels
except (ModuleNotFoundError, ImportError):
    pass
