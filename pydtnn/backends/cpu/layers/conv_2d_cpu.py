import sys

from pydtnn.backends.cpu.layers import LayerCPU
from pydtnn.backends.cpu.layers.conv_2d_variants.best_of_variant import BestOfVariant
from pydtnn.backends.cpu.layers.conv_2d_variants.conv_gemm_variant import ConvGemmVariant
from pydtnn.backends.cpu.layers.conv_2d_variants.depthwise_variant import DepthwiseVariant
from pydtnn.backends.cpu.layers.conv_2d_variants.pointwise_variant import PointwiseVariant
from pydtnn.performance_models import im2col_time, matmul_time, col2im_time
from pydtnn.utils.tensor import PYDTNN_TENSOR_FORMAT
from pydtnn.utils.best_of import BestOf
from pydtnn.utils.best_transpose_0231 import best_transpose_0231
from pydtnn.utils.best_transpose_0312 import best_transpose_0312
from pydtnn.utils.best_transpose_1023 import best_transpose_1023
from pydtnn.utils.memory_cache import MemoryCache

from numpy import ndarray, empty, zeros
from pydtnn.utils.types import shape_t


class Conv2DCPU(LayerCPU,
                DepthwiseVariant,
                PointwiseVariant,
                # I2CVariant (provided from ConvWinogradVariant)
                ConvGemmVariant,
                # ConvWinogradVariant (provided from BestOfVariant)
                # ConvDirectVariant (provided from BestOfVariant)
                BestOfVariant):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Other parameters initialized in initialize()
        self.variant = None
        self.weights = None
        self.biases = None
        self.fwd_time = None
        self.bwd_time = None

    def initialize_i2c(self) -> None:
        # dim_n: Dimension where the "n" of NCHW/NHWC is used in the calculations.
        # self.dim_c: Dimension where the "c" of NCHW/NHWC is used in the calculations.
        dim_n = self.model.batch_size * self.ho * self.wo
        self.dim_c = self.ci * self.kh * self.kw
        match self.model.tensor_format:
            case PYDTNN_TENSOR_FORMAT.NCHW:
                self._x_cols = zeros(shape=(self.dim_c, dim_n), dtype=self.model.dtype, order="C")
                self.res = empty(shape=(self.co, dim_n), dtype=self.model.dtype, order="C")
                self._dw = empty(shape=(self.co, self.dim_c), dtype=self.model.dtype, order="C")
                self.res_bw = empty(shape=(self.dim_c, dim_n), dtype=self.model.dtype, order="C")
            case PYDTNN_TENSOR_FORMAT.NHWC:
                self._x_rows = zeros(shape=(dim_n, self.dim_c), dtype=self.model.dtype, order="C")
                self.res = empty(shape=(dim_n, self.co), dtype=self.model.dtype, order="C")
                self._dw = empty(shape=(self.dim_c, self.co), dtype=self.model.dtype, order="C")
                self.res_bw = empty(shape=(dim_n, self.dim_c), dtype=self.model.dtype, order="C")
            case _:
                raise not NotImplementedError(f"\"{self.model.tensor_format}\" format not implemented.")
        #  NOTE: This is necessary for the initial "reduce_weights_async"
        self.dw: ndarray = zeros(self.weights.shape, dtype=self.model.dtype, order="C")
    # ---

    def initialize_depthwise(self):
        self.dw = zeros(self.weights_shape, dtype=self.model.dtype, order="C")
    # ---

    def initialize_pointwise(self):

        self.dw = empty(shape=self.weights_shape, dtype=self.model.dtype, order="C")
        match self.model.tensor_format:
            case PYDTNN_TENSOR_FORMAT.NCHW:
                self.y = empty(shape=(self.model.batch_size, self.co, self.ho, self.wo), dtype=self.model.dtype, order="C")
                self.dx = empty(shape=(self.ci, self.model.batch_size * self.hi * self.wi), dtype=self.model.dtype, order="C")
            case PYDTNN_TENSOR_FORMAT.NHWC:
                self.y = empty(shape=(self.model.batch_size, self.ho, self.wo, self.co), dtype=self.model.dtype, order="C")
                self.dx = empty(shape=(self.ci, self.model.batch_size * self.hi * self.wi), dtype=self.model.dtype, order="C")
            case _:
                raise NotImplementedError(f"\"DepthwiseVariant\" does not support \"{self.model.tensor_format}\" format.")
    # ---

    def initialize(self, prev_shape: shape_t, x: ndarray | None = None) -> None:
        super().initialize(prev_shape, x)
        # Weights
        self.weights = self.weights_initializer(self.weights_shape, self.model.dtype)
        # Biases
        if self.use_bias:
            self.biases = self.biases_initializer((self.co,), self.model.dtype)
            self.db = empty(shape=(self.co, ), dtype=self.model.dtype, order="C")
        # Select variants if it has not been already selected (e.g., by BestOfVariant)
        if self.variant is None:
            # Select variant when best_of is not enabled
            variant = Conv2DCPU.Variant.I2C  # Default Convolution variant.
            match self.grouping:
                case Conv2DCPU.Grouping.POINTWISE:
                    variant = Conv2DCPU.Variant.POINTWISE
                case Conv2DCPU.Grouping.DEPTHWISE:
                    variant = Conv2DCPU.Variant.DEPTHWISE
                case convWinograd_or_Gemm_or_Direct:
                    # Check colliding options
                    # -> WINOGRAD:
                    if self.model.enable_conv_winograd:
                        if self.model.enable_conv_direct:
                            sys.stderr.write("Error: please, select exactly one of conv_winograd or conv_direct")
                            sys.exit(1)
                        elif self.cw_constraints_fulfilled:
                            variant = Conv2DCPU.Variant.WINOGRAD
                        # else: variant = None # Value set before the match-case statement
                    if variant == Conv2DCPU.Variant.I2C:
                        # assert not self.model.enable_conv_winograd or (self.model.enable_conv_winograd and not self.cw_constraints_fulfilled)
                        # -> GEMM or ConvDirect:
                        if self.model.enable_conv_gemm:
                            # -> GEMM:
                            if self.model.enable_conv_direct:
                                sys.stderr.write("Error: please, select exactly one of conv_gemm or conv_direct")
                                sys.exit(1)
                            else:
                                variant = Conv2DCPU.Variant.GEMM
                        elif self.model.enable_conv_direct:
                            # -> ConvDirect:
                            variant = Conv2DCPU.Variant.DIRECT
                        # else: variant = Conv2DCPU.Variant.I2C # Already set.
            self.variant = variant

        match self.variant:
            case Conv2DCPU.Variant.I2C:
                self.initialize_i2c()
            case Conv2DCPU.Variant.DEPTHWISE:
                self.initialize_depthwise()
            case Conv2DCPU.Variant.POINTWISE:
                self.initialize_pointwise()
            case _:
                pass

        # Set forward and backward implementations based on self.variant
        self.forward, self.backward = self._get_forward_and_backward(self.variant)
        # Performance models
        self.fwd_time = \
            im2col_time(m=(self.ci * self.kh * self.kw), n=(self.model.batch_size * self.ho * self.wo),
                        cpu_speed=self.model.cpu_speed, memory_bw=self.model.memory_bw,
                        dtype=self.model.dtype) + \
            matmul_time(m=self.co, n=(self.model.batch_size * self.ho * self.wo), k=(self.ci * self.kh * self.kw),
                        cpu_speed=self.model.cpu_speed, memory_bw=self.model.memory_bw,
                        dtype=self.model.dtype)
        self.bwd_time = \
            matmul_time(m=self.co, n=(self.ci * self.kh * self.kw), k=(self.model.batch_size * self.ho * self.wo),
                        cpu_speed=self.model.cpu_speed, memory_bw=self.model.memory_bw,
                        dtype=self.model.dtype)
        self.bwd_time += matmul_time(m=(self.ci * self.kh * self.kw), n=(self.model.batch_size * self.ho * self.wo),
                                     k=self.co, cpu_speed=self.model.cpu_speed,
                                     memory_bw=self.model.memory_bw, dtype=self.model.dtype)

    def forward(self, x: ndarray) -> ndarray:
        msg = """This is a fake forward function. It must be masked on initialization by a _forward implementation"""
        NotImplementedError(f"Conv2DCPU forward: {msg}")

    def backward(self, dy: ndarray) -> ndarray:
        msg = """This is a fake backward function. It must be masked on initialization by a _backward implementation"""
        NotImplementedError(f"Conv2DCPU backward: {msg}")

    def print_in_convdirect_format(self) -> None:
        if self.hstride != 1 or self.vstride != 1:
            return
        # #l kn wo ho t kh kw ci wi hi"
        if self.model.tensor_format is PYDTNN_TENSOR_FORMAT.NCHW:
            ci, hi, wi = self.prev_shape
        else:
            hi, wi, ci = self.prev_shape
        print(self.id, self.co, self.wo, self.ho, self.model.batch_size, self.kh, self.kw, ci, wi, hi, sep="\t")

    def _get_forward_and_backward(self, variant: str):
        tensor_format = self.model.tensor_format
        return (getattr(self, f'_forward_{variant}_{tensor_format}'),
                getattr(self, f'_backward_{variant}_{tensor_format}'))
