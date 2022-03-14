#
#  This file is part of Python Distributed Training of Neural Networks (PyDTNN)
#
#  Copyright (C) 2021-22 Universitat Jaume I
#
#  PyDTNN is free software: you can redistribute it and/or modify it under the
#  terms of the GNU General Public License as published by the Free Software
#  Foundation, either version 3 of the License, or (at your option) any later
#  version.
#
#  This program is distributed in the hope that it will be useful, but WITHOUT
#  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
#  or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
#  License for more details.
#
#  You should have received a copy of the GNU General Public License along
#  with this program. If not, see <https://www.gnu.org/licenses/>.
#

from abc import ABC
from typing import Optional, Callable, List

from pydtnn.backends.cpu.layers.conv_2d_variants.conv_direct_variant import ConvDirectVariant
from pydtnn.backends.cpu.layers.conv_2d_variants.conv_winograd_variant import ConvWinogradVariant
from pydtnn.model import TRAIN_MODE, EVALUATE_MODE
from pydtnn.utils import PYDTNN_TENSOR_FORMAT_NCHW
from pydtnn.utils.best_of import BestOf


class BestOfVariant(ConvWinogradVariant, ConvDirectVariant, ABC):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # best_of related attributes (will be initialized in initialize())
        self._best_fw: Optional[BestOf] = None
        self._best_fw_bw_pipeline: Optional[BestOf] = None
        # Other parameters
        self.variant = None

    def initialize(self, prev_shape, need_dx=True):
        super().initialize(prev_shape, need_dx)
        if self.model.enable_best_of:
            # Set variant to 'best_of' and set alternatives to only forward, and forward backward best_ofs
            self.variant = 'best_of'
            # Bestof will honor the next configuration options:
            # - enable_conv_winograd
            # - enable_conv_gemm
            # - enable_conv_direct
            # - conv_direct_methods_for_best_of (if empty, conv_direct_method will be used instead)
            # Set alternatives for only forward, and for forward backward
            alternatives_fw = []
            alternatives_fw_bw_pipeline = []
            if self.model.enable_conv_i2c:
                alternatives_fw.append(('i2c', self._get_class_forward_and_backward('i2c')[0]))
                alternatives_fw_bw_pipeline.append(('i2c', self._get_class_forward_and_backward('i2c')))
            if self.model.enable_conv_gemm:
                alternatives_fw.append(('cg', self._get_class_forward_and_backward('cg')[0]))
                alternatives_fw_bw_pipeline.append(('cg', self._get_class_forward_and_backward('cg')))
            if self.model.enable_conv_winograd and self.cw_constraints_fulfilled:
                alternatives_fw.append(('cw', self._get_class_forward_and_backward('cw')[0]))
                alternatives_fw_bw_pipeline.append(('cw', self._get_class_forward_and_backward('cw')))
            if self.model.enable_conv_direct:
                for n in range(len(self.cd)):
                    cdn = f"cd{n}"
                    alternatives_fw.append((cdn, self._get_class_forward_and_backward(cdn)[0]))
                    alternatives_fw_bw_pipeline.append((cdn, self._get_class_forward_and_backward(cdn)))
            self._best_fw = BestOf(
                name="Conv2DCPU only forward",
                alternatives=alternatives_fw,
                get_problem_size=lambda *args: tuple(list(args[0].shape) + list(args[0].weights.shape)),
            )
            self._best_fw_bw_pipeline = BestOf(
                name="Conv2DCPU forward backward",
                alternatives=alternatives_fw_bw_pipeline,
                get_problem_size=lambda *args: tuple(list(args[0].shape) + list(args[0].weights.shape)),
            )

    def _get_class_forward_and_backward(self, variant) -> List[Callable]:
        tensor_format = 'nchw' if self.model.tensor_format == PYDTNN_TENSOR_FORMAT_NCHW else 'nhwc'
        return [getattr(self.__class__, f'_forward_{variant}_{tensor_format}'),
                getattr(self.__class__, f'_backward_{variant}_{tensor_format}')]

    def _fw_bw_best_of(self, stage, x_or_y):
        if self.model.mode == TRAIN_MODE:
            # noinspection PyTypeChecker
            return self._best_fw_bw_pipeline(stage, self, x_or_y)
        elif self.model.mode == EVALUATE_MODE:
            # noinspection PyTypeChecker
            return self._best_fw(self, x_or_y)
        else:
            raise RuntimeError("Conv2D BestOf variant requires Model.mode to be set to EVALUATE_MODE or TRAIN_MODE")

    def _forward_best_of_nhwc(self, x):
        return self._fw_bw_best_of(0, x)

    def _forward_best_of_nchw(self, x):
        return self._fw_bw_best_of(0, x)

    def _backward_best_of_nhwc(self, y):
        return self._fw_bw_best_of(1, y)

    def _backward_best_of_nchw(self, y):
        return self._fw_bw_best_of(1, y)
