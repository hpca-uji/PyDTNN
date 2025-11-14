import sys
from warnings import warn

from pydtnn.layers.conv_2d import Conv2D
from pydtnn.backends.cpu.layers.layer_cpu import LayerCPU
from pydtnn.backends.cpu.layers.conv_2d_variants.best_of_variant import BestOfVariant
from pydtnn.backends.cpu.layers.conv_2d_variants.conv_gemm_variant import ConvGemmVariant
from pydtnn.backends.cpu.layers.conv_2d_variants.depthwise_variant import DepthwiseVariant
from pydtnn.backends.cpu.layers.conv_2d_variants.pointwise_variant import PointwiseVariant
from pydtnn.utils.performance_models import im2col_time, matmul_time
from pydtnn.utils.tensor import SampleFormat, TensorFormat, format_transpose
from pydtnn.utils.types import ArrayShape

import numpy as np


class Conv2DCPU(LayerCPU,
                DepthwiseVariant[np.ndarray],
                PointwiseVariant[np.ndarray],
                # I2CVariant (provided from ConvWinogradVariant)
                ConvGemmVariant[np.ndarray],
                # ConvWinogradVariant (provided from BestOfVariant)
                # ConvDirectVariant (provided from BestOfVariant)
                BestOfVariant[np.ndarray],
                Conv2D[np.ndarray]
                ):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # More parameters initialized in initialize()
        self.variant = None
        self.biases = None  # type: ignore
        self.weights = None  # type: ignore
        self.fwd_time = None  # type: ignore
        self.bwd_time = None  # type: ignore

    def _export(self) -> dict:
        data = super()._export()

        match self.model.tensor_format:
            case TensorFormat.NHWC:
                match self.grouping:
                    case Conv2D.Grouping.POINTWISE:
                        # NHWC's src: ci, co
                        # NCHW's dst: co, ci
                        data["weights"] = format_transpose(data["weights"], "IO", "OI")
                    case Conv2D.Grouping.STANDARD:
                        # NHWC's src: ci, kh, kw, co
                        # NCHW's dst: co, ci, kh, kw
                        data["weights"] = format_transpose(data["weights"], "IHWO", "OIHW")

        return data

    def _import(self, data) -> None:
        match self.model.tensor_format:
            case TensorFormat.NHWC:
                match self.grouping:
                    case Conv2D.Grouping.POINTWISE:
                        # NCHW's src: co, ci
                        # NHWC's dst: ci, co
                        data["weights"] = format_transpose(data["weights"], "OI", "IO")
                    case Conv2D.Grouping.STANDARD:
                        # NCHW's src: co, ci, kh, kw
                        # NHWC's dst: ci, kh, kw, co
                        data["weights"] = format_transpose(data["weights"], "OIHW", "IHWO")

        super()._import(data)

    def initialize_i2c(self) -> None:
        # self.dim_n: Dimension where the "n" of NCHW/NHWC is used in the calculations.
        # self.dim_c: Dimension where the "c" of NCHW/NHWC is used in the calculations.
        self.dim_n = self.model.batch_size * self.ho * self.wo
        self.dim_c = self.ci * self.kh * self.kw
        match self.model.tensor_format:
            case TensorFormat.NCHW:
                self._x_cols = np.zeros(shape=(self.dim_c, self.dim_n), dtype=self.model.dtype, order="C")
                # NOTE: These attributes only store data, their values before the operation doesn't matter; they're initalized due avoid warnings in "LayerAndActivationBase.export".
                self.res = np.zeros(shape=(self.co, self.dim_n), dtype=self.model.dtype, order="C")
                self._dw = np.zeros(shape=(self.co, self.dim_c), dtype=self.model.dtype, order="C")
                self.res_bw = np.zeros(shape=(self.dim_c, self.dim_n), dtype=self.model.dtype, order="C")
            case TensorFormat.NHWC:
                self._x_rows = np.zeros(shape=(self.dim_n, self.dim_c), dtype=self.model.dtype, order="C")
                # NOTE: These attributes only store data, their values before the operation doesn't matter; they're initalized due avoid warnings in "LayerAndActivationBase.export".
                self.res = np.zeros(shape=(self.dim_n, self.co), dtype=self.model.dtype, order="C")
                self._dw = np.zeros(shape=(self.dim_c, self.co), dtype=self.model.dtype, order="C")
                self.res_bw = np.zeros(shape=(self.dim_n, self.dim_c), dtype=self.model.dtype, order="C")
            case _:
                raise NotImplementedError(f"\"{self.model.tensor_format}\" format not implemented.")
        #  NOTE: This is necessary for the initial "reduce_weights_async"
        self.dw: np.ndarray = np.zeros(self.weights.shape, dtype=self.model.dtype, order="C")
    # ---

    def initialize_depthwise(self):
        self.dw = np.zeros(self.weights_shape, dtype=self.model.dtype, order="C")
    # ---

    def initialize_pointwise(self):

        y_shape = self.model.encode_shape((self.model.batch_size, self.co, self.ho, self.wo))
        
        # NOTE: These attributes only store data, their values before the operation doesn't matter; they're initalized due avoid warnings in "LayerAndActivationBase.export".
        self.dw = np.zeros(shape=self.weights_shape, dtype=self.model.dtype, order="C")
        self.y = np.zeros(shape=y_shape, dtype=self.model.dtype, order="C")

        match self.model.tensor_format:
            case TensorFormat.NCHW:
                # NOTE: This attribute only stores data, its values before the operation doesn't matters; it's initalized due avoid warnings in "LayerAndActivationBase.export".
                self.dx = np.zeros(shape=(self.ci, self.model.batch_size * self.hi * self.wi), dtype=self.model.dtype, order="C")
            case TensorFormat.NHWC:
                # NOTE: This attribute only stores data, its values before the operation doesn't matters; it's initalized due avoid warnings in "LayerAndActivationBase.export".
                self.dx = np.zeros(shape=(self.ci, self.model.batch_size * self.hi * self.wi), dtype=self.model.dtype, order="C")
            case _:
                raise NotImplementedError(f"\"DepthwiseVariant\" does not support \"{self.model.tensor_format}\" format.")
    # ---

    def initialize(self, prev_shape: ArrayShape, x: np.ndarray | None = None) -> None:
        super().initialize(prev_shape, x)
        # Weights
        self.weights = self.weights_initializer(self.weights_shape, self.model.dtype)
        # Select variants if it has not been already selected (e.g., by BestOfVariant)
        bias_shape = (self.co,)

        if self.variant is None:
            # Select variant when best_of is not enabled
            variant = Conv2DCPU.Variant.I2C  # Default Convolution variant.
            # bias_shape = (self.co,) # I2C, POINTWISE, DEPTHWISE
            match self.grouping:
                case Conv2DCPU.Grouping.POINTWISE:
                    variant = Conv2DCPU.Variant.POINTWISE
                case Conv2DCPU.Grouping.DEPTHWISE:
                    variant = Conv2DCPU.Variant.DEPTHWISE
                case _:  # convGemm or convWinograd or convDirect
                    # Check colliding options
                    if (self.model.enable_conv_gemm + self.model.enable_conv_winograd + self.model.enable_conv_direct) > 1:
                        raise ValueError("Select exactly one of convGemm or convWinograd or convDirect")

                    if self.model.enable_conv_gemm:
                        variant = Conv2DCPU.Variant.GEMM
                        # TODO: Change this.
                        bias_shape = self.model.encode_shape((self.model.batch_size, self.co, self.ho, self.wo))

                    elif self.model.enable_conv_winograd:
                        if self.cw_constraints_fulfilled:
                            variant = Conv2DCPU.Variant.WINOGRAD
                            # bias_shape = (self.co,)
                        else:
                            warn("Winograd constraints not fulfilled, using fallback!")

                    elif self.model.enable_conv_direct:
                        variant = Conv2DCPU.Variant.DIRECT
                        # TODO: Change this.
                        bias_shape = self.model.encode_shape((self.model.batch_size, self.co, self.ho, self.wo))

            self.variant = variant

        # Biases
        if self.use_bias:
            self.biases = self.biases_initializer(bias_shape, self.model.dtype)
            self.db = np.zeros(shape=bias_shape, dtype=self.model.dtype, order="C")

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
                        dtype=self.model.dtype)  # type: ignore (It works well.)
        self.bwd_time = \
            matmul_time(m=self.co, n=(self.ci * self.kh * self.kw), k=(self.model.batch_size * self.ho * self.wo),
                        cpu_speed=self.model.cpu_speed, memory_bw=self.model.memory_bw,
                        dtype=self.model.dtype)
        self.bwd_time += matmul_time(m=(self.ci * self.kh * self.kw), n=(self.model.batch_size * self.ho * self.wo),
                                     k=self.co, cpu_speed=self.model.cpu_speed,
                                     memory_bw=self.model.memory_bw, dtype=self.model.dtype)

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Use a real forward variant!")

    def backward(self, dy: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Use a real backwards variant!")

    def print_in_convdirect_format(self) -> None:
        if self.hstride != 1 or self.vstride != 1:
            return
        # #l kn wo ho t kh kw ci wi hi"
        ci, hi, wi = self.model.decode_shape(self.prev_shape)
        print(self.id, self.co, self.wo, self.ho, self.model.batch_size, self.kh, self.kw, ci, wi, hi, sep="\t")

    def _get_forward_and_backward(self, variant: str):
        tensor_format = self.model.tensor_format
        return (getattr(self, f'_forward_{variant}_{tensor_format}'),
                getattr(self, f'_backward_{variant}_{tensor_format}'))

    @property
    def canonical_name(self) -> str:
        return f"{super().canonical_name}_{self.variant}"
