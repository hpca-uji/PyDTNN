"""
Tracer definitions for Python Distributed Training of Neural Networks (PyDTNN)

PyDTNN is a light-weight library for distributed Deep Learning training and
inference that offers an initial starting point for interaction with distributed
training of (and inference with) deep neural networks. PyDTNN prioritizes
simplicity over efficiency, providing an amiable user interface which enables a
flat accessing curve. To perform the training and inference processes, PyDTNN
exploits distributed inter-process parallelism (via MPI) for clusters and
intra-process (via multi-threading) parallelism to leverage the presence of
multicore processors and GPUs at node level. For that, PyDTNN uses MPI4Py for
message-passing, BLAS calls via NumPy for multicore processors and
PyCUDA+cuDNN+cuBLAS for NVIDIA GPUs.

Copyright 2020 Universitat Jaume I

This file is part of PyDTNN. PyDTNN is free software: you can redistribute it
and/or modify it under the terms of the GNU General Public License as published
by the Free Software Foundation, either version 3 of the License, or (at your
option) any later version.

PyDTNN is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details. You
should have received a copy of the GNU General Public License along with this
program. If not, see <http://www.gnu.org/licenses/>.
"""

__author__ = "Manuel F. Dolz, Enrique S. Quintana, \
              Mar Catalan, Adrian Castello"
__contact__ = "dolzm@uji.es"
__copyright__ = "Copyright 2020, Universitat Jaume I"
__credits__ = ["Manuel F. Dolz, Enrique S. Quintana", \
               "Mar Catalan", "Adrian Castello"]
__date__ = "2020/03/22"

__email__ = "dolzm@uji.es"
__license__ = "GPLv3"
__maintainer__ = "Manuel F. Dolz"
__status__ = "Production"
__version__ = "1.1.0"

import ctypes
import os
from importlib import import_module

# ---
PYDTNN_MDL_EVENT = 60000001
PYDTNN_MDL_FORWARD = 1
PYDTNN_MDL_BACKWARD = 2
PYDTNN_MDL_ALLREDUCE_DW = 3
PYDTNN_MDL_WAIT_DW = 4
PYDTNN_MDL_UPDATE_DW = 5
PYDTNN_MDL_EVENTS = 5
# ---
PYDTNN_OPS_EVENT = 60000002
PYDTNN_OPS_FORWARD_MATMUL = 1
PYDTNN_OPS_FORWARD_IM2COL = 2
PYDTNN_OPS_COMP_DX_MATMUL = 3
PYDTNN_OPS_COMP_DX_COL2IM = 4
PYDTNN_OPS_COMP_DW_MATMUL = 5
PYDTNN_OPS_ALLREDUCE_DW = 6
PYDTNN_OPS_EVENTS = 6
# ---


class Tracer:

    def __init__(self, tracing=False):
        self.tracing = tracing
        self.mdl_events = {}
        self.ops_events = {}
        if not self.tracing:
            self.define_event_type = self._do_nothing
            self.emit_event = self._do_nothing
            self.emit_nevent = self._do_nothing

    def define_event_type(self, model):
        self.mdl_events = {0: "End"}
        self.ops_events = {0: "End"}
        for i in range(len(model.layers)):
            layer_name = type(model.layers[i]).__name__
            self.mdl_events[i * PYDTNN_MDL_EVENTS + PYDTNN_MDL_FORWARD] = f"{i}_{layer_name}_forward "
            self.mdl_events[i * PYDTNN_MDL_EVENTS + PYDTNN_MDL_BACKWARD] = f"{i}_{layer_name}_backward "
            self.mdl_events[i * PYDTNN_MDL_EVENTS + PYDTNN_MDL_ALLREDUCE_DW] = f"{i}_{layer_name}_allreduce_dw "
            self.mdl_events[i * PYDTNN_MDL_EVENTS + PYDTNN_MDL_WAIT_DW] = f"{i}_{layer_name}_wait_dw "
            self.mdl_events[i * PYDTNN_MDL_EVENTS + PYDTNN_MDL_UPDATE_DW] = f"{i}_{layer_name}_update_dw "
            self.ops_events[i * PYDTNN_OPS_EVENTS + PYDTNN_OPS_FORWARD_MATMUL] = f"{i}_{layer_name}_forward_matmul "
            self.ops_events[i * PYDTNN_OPS_EVENTS + PYDTNN_OPS_FORWARD_IM2COL] = f"{i}_{layer_name}_forward_im2col "
            self.ops_events[i * PYDTNN_OPS_EVENTS + PYDTNN_OPS_COMP_DX_MATMUL] = f"{i}_{layer_name}_compute_dx_matmul "
            self.ops_events[i * PYDTNN_OPS_EVENTS + PYDTNN_OPS_COMP_DX_COL2IM] = f"{i}_{layer_name}_compute_dx_col2im "
            self.ops_events[i * PYDTNN_OPS_EVENTS + PYDTNN_OPS_COMP_DW_MATMUL] = f"{i}_{layer_name}_compute_dw_matmul "
            self.ops_events[i * PYDTNN_OPS_EVENTS + PYDTNN_OPS_ALLREDUCE_DW] = f"{i}_{layer_name}_allreduce_dw "

    def emit_event(self, evt, val):
        pass

    def emit_nevent(self, evt, val):
        pass

    def _do_nothing(self, *args, **kwargs):
        pass


class ExtraeTracer(Tracer):

    def __init__(self, tracing=False):
        super().__init__(tracing)
        if self.tracing:
            self.pyextrae = import_module('pyextrae.common.extrae')

    def define_event_type(self, model):
        super().define_event_type(model)
        n_values = len(model.layers) * PYDTNN_MDL_EVENTS + 1
        description = "Model layers"
        codes = (ctypes.c_ulonglong * n_values)()
        descriptions = (ctypes.c_char_p * n_values)()
        for code, description in self.mdl_events.items():
            codes[code] = code
            descriptions[code] = description
        self.pyextrae.Extrae[os.getpid()].Extrae_define_event_type(
            ctypes.pointer(ctypes.c_uint(PYDTNN_MDL_EVENT)),
            ctypes.c_char_p(description.encode('utf-8')),
            ctypes.pointer(ctypes.c_uint(n_values)),
            ctypes.pointer(codes),
            ctypes.pointer(descriptions))
        n_values = len(model.layers) * PYDTNN_OPS_EVENTS + 1
        description = "PyDTNN ops per layer"
        codes = (ctypes.c_ulonglong * n_values)()
        descriptions = (ctypes.c_char_p * n_values)()
        for code, description in self.ops_events.items():
            codes[code] = code
            descriptions[code] = description
        self.pyextrae.Extrae[os.getpid()].Extrae_define_event_type(
            ctypes.pointer(ctypes.c_uint(PYDTNN_OPS_EVENT)),
            ctypes.c_char_p(description.encode('utf-8')),
            ctypes.pointer(ctypes.c_uint(n_values)),
            ctypes.pointer(codes),
            ctypes.pointer(descriptions))

    def emit_event(self, evt, val):
        self.pyextrae.eventandcounters(evt, val)

    def emit_nevent(self, evt, val):
        self.pyextrae.neventandcounters(evt, val)
