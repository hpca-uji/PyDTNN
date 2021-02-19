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

__author__ = "Manuel F. Dolz, Enrique S. Quintana, Sergio Barrachina, Mar Catalán, Adrián Castelló"
__contact__ = "dolzm@uji.es"
__copyright__ = "Copyright 2021, Universitat Jaume I"
__credits__ = ["Manuel F. Dolz, Enrique S. Quintana", "Sergio Barrachina", "Mar Catalán", "Adrián Castelló"]
__date__ = "2020/03/22"

__email__ = "dolzm@uji.es"
__license__ = "GPLv3"
__maintainer__ = "Manuel F. Dolz"
__status__ = "Production"
__version__ = "1.1.0"

import atexit
import ctypes
import os
import resource
import sys

from collections import defaultdict
from importlib import import_module
from timeit import default_timer as timer

# ---
PYDTNN_MDL_EVENT = 60000001
PYDTNN_MDL_EVENTS = 5
(PYDTNN_MDL_FORWARD,
 PYDTNN_MDL_BACKWARD,
 PYDTNN_MDL_ALLREDUCE_DW,
 PYDTNN_MDL_WAIT_DW,
 PYDTNN_MDL_UPDATE_DW) = range(1, PYDTNN_MDL_EVENTS + 1)
# ---
PYDTNN_OPS_EVENT = 60000002
PYDTNN_OPS_EVENTS = 28
(PYDTNN_OPS_FORWARD_IM2COL,
 PYDTNN_OPS_FORWARD_RESHAPE_W,
 PYDTNN_OPS_FORWARD_MATMUL,
 PYDTNN_OPS_FORWARD_SUM_BIASES,
 PYDTNN_OPS_FORWARD_RESHAPE_Y,
 PYDTNN_OPS_FORWARD_CONVGEMM,
 PYDTNN_OPS_COMP_DX_MATMUL,
 PYDTNN_OPS_COMP_DX_COL2IM,
 PYDTNN_OPS_COMP_DW_MATMUL,
 PYDTNN_OPS_ALLREDUCE_DW,
 PYDTNN_OPS_BACKWARD_DECONV_GEMM,
 PYDTNN_OPS_BACKWARD_DCG_TRANSPOSE_DY,
 PYDTNN_OPS_BACKWARD_DCG_SHRINK,
 PYDTNN_OPS_BACKWARD_TRANSPOSE_DY,
 PYDTNN_OPS_BACKWARD_TRANSPOSE_W,
 PYDTNN_OPS_BACKWARD_IM2COL,
 PYDTNN_OPS_BACKWARD_PADDING_X,
 PYDTNN_OPS_BACKWARD_COMP_NEW_INDEXES,
 PYDTNN_OPS_BACKWARD_REINDEX,
 PYDTNN_OPS_BACKWARD_CONVGEMM,
 PYDTNN_OPS_CONVGEMM_X_PAD,
 PYDTNN_OPS_CONVGEMM_CG,
 PYDTNN_OPS_CONVGEMM_TRANS_X_PAD,
 PYDTNN_OPS_CONVGEMM_TRANS_CG,
 PYDTNN_OPS_CONVGEMM_TRANS_BIASES,
 PYDTNN_OPS_CONVGEMM_TRANS_TR1230,
 PYDTNN_OPS_BACKWARD_RESHAPE_DW,
 PYDTNN_OPS_BACKWARD_SUM_BIASES) = range(1, PYDTNN_OPS_EVENTS + 1)


class EventType:
    """EventType container"""

    def __init__(self, name):
        self.name = name
        self._events = {}

    def __getitem__(self, item):
        try:
            description = self._events[item]
        except KeyError:
            sys.stderr.write(f"SimpleTracer warning: No event with code '{item}' "
                             f"in the '{self.name}' type of events.\n")
            return f"Unknown code {self.name}"
        return description

    def __setitem__(self, value, description):
        self._events[value] = description

    def __len__(self):
        return len(self._events)

    def items(self):
        return self._events.items()


class Tracer:
    """Base class for tracers"""

    def __init__(self, tracing):
        self.event_types = {
            PYDTNN_MDL_EVENT: EventType("Model"),
            PYDTNN_OPS_EVENT: EventType("Operations"),
        }
        if tracing:
            self.enable_tracing()
            self.enable_print_memory_usage()
        else:
            self.disable_tracing()
            self.disable_print_memory_usage()

    def enable_tracing(self):
        """Actions that must be done if tracing is enabled"""
        setattr(self, "define_event_types", self._define_event_types)
        setattr(self, "emit_event", self._emit_event)
        setattr(self, "emit_nevent", self._emit_nevent)

    def disable_tracing(self):
        """Actions that must be done if tracing is disabled"""
        setattr(self, "define_event_types", lambda *args, **kwargs: None)
        setattr(self, "emit_event", lambda *args, **kwargs: None)
        setattr(self, "emit_nevent", lambda *args, **kwargs: None)

    def enable_print_memory_usage(self):
        """Actions that must be done if print memory usage is enabled"""
        setattr(self, "print_memory_usage", self._print_memory_usage)

    def disable_print_memory_usage(self):
        """Actions that must be done if print memory usage is disabled"""
        setattr(self, "print_memory_usage", lambda *args, **kwargs: None)

    def define_event_types(self, model):
        """Fake method, will be replaced by lambda: None or _define_event_types()"""
        pass

    def emit_event(self, evt_type, evt_val):
        """Fake method, will be replaced by lambda: None or _emit_event()"""
        pass

    def emit_nevent(self, evt_evt, evt_val):
        """Fake method, will be replaced by lambda: None or _emit_nevent()"""
        pass

    def print_memory_usage(self, text):
        """Fake method, will be replaced by lambda: None or _print_memory_usage()"""
        pass

    def _get_layers_recursively(self, layers):
        all_layers = []
        for layer in layers:
            all_layers.append(layer)
            all_layers += self._get_layers_recursively(layer.children)
        return all_layers

    def _define_event_types(self, model):
        """This method will be called only if tracing is enabled"""
        mdl_event = self.event_types[PYDTNN_MDL_EVENT]
        ops_event = self.event_types[PYDTNN_OPS_EVENT]
        mdl_event[0] = "End"
        ops_event[0] = "End"
        constants = dict(globals())  # warning: constants must be a copy of globals()
        for name in ["PYDTNN_MDL_EVENT", "PYDTNN_MDL_EVENTS", "PYDTNN_OPS_EVENT", "PYDTNN_OPS_EVENTS"]:
            constants.pop(name)
        mdl_constants = [(name, val) for name, val in constants.items() if name[:len("PYDTNN_MDL_")] == "PYDTNN_MDL_"]
        ops_constants = [(name, val) for name, val in constants.items() if name[:len("PYDTNN_OPS_")] == "PYDTNN_OPS_"]
        for layer in self._get_layers_recursively(model.layers):
            layer_name = type(layer).__name__
            for (name, val) in mdl_constants:
                mdl_event[layer.id * PYDTNN_MDL_EVENTS + val] = f"{layer.id:03}_{layer_name}_{name[11:].lower()}"
            for (name, val) in ops_constants:
                ops_event[layer.id * PYDTNN_OPS_EVENTS + val] = f"{layer.id:03}_{layer_name}_{name[11:].lower()}"

    def _emit_event(self, evt_type, evt_val):
        """This method will be called only if tracing is enabled"""
        pass

    def _emit_nevent(self, evt_evt, evt_val):
        """This method will be called only if tracing is enabled"""
        pass

    @staticmethod
    def _print_memory_usage(text=""):
        """This method will be called only if print memory usage is enabled"""
        u = resource.getrusage(resource.RUSAGE_SELF)
        if text != "":
            text = f" {text}:"
        print(f">>>{text} user time={u[0]:.2f}, sys time={u[1]:.2f}, mem={u[2] / 1024:.2f} MiB")


class ExtraeTracer(Tracer):

    def __init__(self, tracing):
        super().__init__(tracing)
        self.pyextrae = None  # Declared here, will be initialized on enable_tracing()

    def enable_tracing(self):
        super().enable_tracing()
        self.pyextrae = import_module('pyextrae.common.extrae')

    def _define_event_types(self, model):
        """This method will be called only if tracing is enabled"""
        super()._define_event_types(model)
        for event_type_value, event_type in self.event_types.items():
            description = event_type.name
            nvalues = len(event_type)
            values = (ctypes.c_ulonglong * nvalues)()
            descriptions = (ctypes.c_char_p * nvalues)()
            for i, description in event_type.items():
                values[i] = i
                descriptions[i] = description
            self.pyextrae.Extrae[os.getpid()].Extrae_define_event_type(
                ctypes.pointer(ctypes.c_uint(event_type_value)),
                ctypes.c_char_p(description.encode('utf-8')),
                ctypes.pointer(ctypes.c_uint(nvalues)),
                ctypes.pointer(values),
                ctypes.pointer(descriptions))

    def _emit_event(self, evt_type, val):
        """This method will be called only if tracing is enabled"""
        self.pyextrae.eventandcounters(evt_type, val)

    def _emit_nevent(self, evt, val):
        """This method will be called only if tracing is enabled"""
        self.pyextrae.neventandcounters(evt, val)


class SimpleTracer(Tracer):

    def __init__(self, tracing, output_filename, comm):
        super().__init__(tracing)
        self.output_filename = output_filename
        self.rank = 0
        if comm is not None:
            self.rank = comm.Get_rank()
        self.events = defaultdict(lambda: defaultdict(lambda: [0, 0.0]))
        self.pending_events = []

    def enable_tracing(self):
        super().enable_tracing()
        # If tracing is enabled at least once, register self.write_output to be executed at exit
        atexit.register(self._write_output)

    def _emit_event(self, evt_type_val, evt_val):
        """This method will be called only if tracing is enabled"""
        if evt_val != 0:
            self.pending_events.append((evt_type_val, evt_val, timer()))
        else:
            toc = timer()
            if len(self.pending_events) == 0:
                raise ValueError("Received an 'End' event but there are no pending events!")
            if self.pending_events[-1][0] != evt_type_val:
                raise ValueError("Received an 'End' event for a different event type than expected!")
            _evt_type_val, _evt_val, tic = self.pending_events.pop()
            previous_calls, previous_time = self.events[_evt_type_val][_evt_val]
            self.events[_evt_type_val][_evt_val] = [previous_calls + 1, previous_time + toc - tic]

    def _emit_nevent(self, evt_type_val_list, evt_val_list):
        """This method will be called only if tracing is enabled"""
        zipped_list = list(zip(evt_type_val_list, evt_val_list))
        if evt_val_list[0] == 0:
            zipped_list = reversed(zipped_list)
        for evt_type_val, evt_val in zipped_list:
            self.emit_event(evt_type_val, evt_val)

    def _write_output(self):
        """This method will be called at exit only if tracing has been enabled at any time"""
        if self.rank == 0:
            if len(self.pending_events):
                print("Warning: finishing simple tracer while there are pending events to be marked as finished.")
            print(f"Writing simple tracer output to '{self.output_filename}'...")
            with open(self.output_filename, 'w') as f:
                f.write("Event type;Event value;Event name;Calls;Total time;Time per call\n")
                for event_type_key, events in self.events.items():
                    event_type = self.event_types[event_type_key]
                    event_type_name = event_type.name
                    for value, (_calls, _time) in events.items():
                        f.write(f"{event_type_name};{value};{event_type[value]};{_calls};{_time};{_time / _calls}\n")
