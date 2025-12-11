from typing import Callable, List

from pydtnn.backends.cpu.layers.conv_2d.direct_cpu import Conv2DDirectCPU
from pydtnn.backends.cpu.layers.conv_2d.winograd_cpu import Conv2DWinogradCPU
from pydtnn.model import Model
from pydtnn.utils.best_of import BestOf

import numpy as np
from pydtnn.utils.constants import ArrayShape



# FIXME: Broken since Conv2D to backend support
class BestOfVariant(Conv2DWinogradCPU, Conv2DDirectCPU):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # best_of related attributes (will be initialized in initialize())
        self._best_fw: BestOf = None # type: ignore
        self._best_fw_bw_pipeline: BestOf = None # type: ignore
        # Other parameters
        self.variant = None

    def initialize(self, prev_shape: ArrayShape, x: np.ndarray | None = None):
        super().initialize(prev_shape, x)
        if self.model.enable_best_of:
            # Set variant to 'best_of' and set alternatives to only forward, and forward backward best_ofs
            self.variant = Conv2DDirectCPU.Variant.BEST_OF
            # Bestof will honor the next configuration options:
            # - enable_conv_winograd
            # - enable_conv_gemm
            # - enable_conv_direct
            # - conv_direct_methods_for_best_of (if empty, conv_direct_method will be used instead)
            # Set alternatives for only forward, and for forward backward
            alternatives_fw = []
            alternatives_fw_bw_pipeline = []
            if self.model.enable_conv_i2c:
                alternatives_fw.append((Conv2DDirectCPU.Variant.I2C, self._get_class_forward_and_backward(Conv2DDirectCPU.Variant.I2C)[0]))
                alternatives_fw_bw_pipeline.append((Conv2DDirectCPU.Variant.I2C, self._get_class_forward_and_backward(Conv2DDirectCPU.Variant.I2C)))
            if self.model.enable_conv_gemm:
                alternatives_fw.append((Conv2DDirectCPU.Variant.GEMM, self._get_class_forward_and_backward(Conv2DDirectCPU.Variant.GEMM)[0]))
                alternatives_fw_bw_pipeline.append((Conv2DDirectCPU.Variant.GEMM, self._get_class_forward_and_backward(Conv2DDirectCPU.Variant.GEMM)))
            if self.model.enable_conv_winograd and self.cw_constraints_fulfilled:
                alternatives_fw.append((Conv2DDirectCPU.Variant.WINOGRAD, self._get_class_forward_and_backward(Conv2DDirectCPU.Variant.WINOGRAD)[0]))
                alternatives_fw_bw_pipeline.append((Conv2DDirectCPU.Variant.WINOGRAD, self._get_class_forward_and_backward(Conv2DDirectCPU.Variant.WINOGRAD)))
            if self.model.enable_conv_direct:
                for n in range(len(self.cd)):
                    cdn = f"cd{n}"
                    alternatives_fw.append((cdn, self._get_class_forward_and_backward(cdn)[0]))
                    alternatives_fw_bw_pipeline.append((cdn, self._get_class_forward_and_backward(cdn)))
            self._best_fw = BestOf(
                name="Conv2DCPU only forward",
                alternatives=alternatives_fw,
                get_problem_size=lambda *args: tuple(args[0].shape) + tuple(args[0].weights.shape)
                + (args[0].vstride,
                   args[0].hstride,
                   args[0].vdilation,
                   args[0].hdilation),
            )
            self._best_fw_bw_pipeline = BestOf(
                name="Conv2DCPU forward backward",
                alternatives=alternatives_fw_bw_pipeline,
                get_problem_size=lambda *args: tuple(args[0].shape) + tuple(args[0].weights.shape)
                + (args[0].vpadding, args[0].hpadding,
                   args[0].vstride, args[0].hstride,
                   args[0].vdilation, args[0].hdilation),
            )

    def _get_class_forward_and_backward(self, variant) -> List[Callable]:
        return [getattr(self.__class__, f'_forward_{variant}_{self.model.tensor_format}'),
                getattr(self.__class__, f'_backward_{variant}_{self.model.tensor_format}')]

    def _fw_bw_best_of(self, stage, x_or_y):
        match self.model.mode:
            case Model.Mode.TRAIN:
                return self._best_fw_bw_pipeline(stage, self, x_or_y)
            case Model.Mode.EVALUATE:
                return self._best_fw(self, x_or_y)
            case _:
                raise RuntimeError("Conv2D BestOf variant requires Model.mode to be set to ModelModeEnum.EVALUATE or ModelModeEnum.TRAIN")

    def _forward_best_of_nhwc(self, x):
        return self._fw_bw_best_of(0, x)

    def _forward_best_of_nchw(self, x):
        return self._fw_bw_best_of(0, x)

    def _backward_best_of_nhwc(self, y):
        return self._fw_bw_best_of(1, y)

    def _backward_best_of_nchw(self, y):
        return self._fw_bw_best_of(1, y)
