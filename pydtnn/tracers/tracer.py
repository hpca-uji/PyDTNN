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
#  with this program.  If not, see <https://www.gnu.org/licenses/>.
#

import resource
import sys
from abc import abstractmethod

from .events import *


class EventType:
    """
    EventType container
    """

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


class PostInitCaller(type):
    def __call__(cls, *args, **kwargs):
        obj = type.__call__(cls, *args, **kwargs)
        obj.__post_init__()
        return obj


class Tracer(metaclass=PostInitCaller):
    """
    Tracer base class
    """

    def __init__(self, tracing):
        self.event_types = {
            PYDTNN_MDL_EVENT: EventType("Model"),
            PYDTNN_OPS_EVENT: EventType("Operations"),
        }
        self.tracing = tracing

    def __post_init__(self):
        """
        This method will be called AFTER all the derived classes __init__ methods are completed.
        By proceeding in this way, when the derived classes enable/disable methods are called, all the attributes
        they require will already have been defined on their corresponding __init__ methods.
        """
        if self.tracing:
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

    def emit_event(self, evt_type, evt_val, stream=None):
        """Fake method, will be replaced by lambda: None or _emit_event()"""
        pass

    def emit_nevent(self, evt_evt, evt_val, stream=None):
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
        for layer in model.get_all_layers():
            for (name, val) in mdl_constants:
                mdl_event[layer.id * PYDTNN_MDL_EVENTS + val] = f"{layer.canonical_name_with_id}_{name[11:].lower()}"
            for (name, val) in ops_constants:
                ops_event[
                    layer.id * PYDTNN_OPS_EVENTS + val] = f"{layer.id:03}_{layer.canonical_name}_{name[11:].lower()}"

    @abstractmethod
    def _emit_event(self, evt_type, evt_val, stream=None):
        """This method will be called only if tracing is enabled"""
        pass

    @abstractmethod
    def _emit_nevent(self, evt_evt, evt_val, stream=None):
        """This method will be called only if tracing is enabled"""
        pass

    @staticmethod
    def _print_memory_usage(text=""):
        """This method will be called only if print memory usage is enabled"""
        u = resource.getrusage(resource.RUSAGE_SELF)
        if text != "":
            text = f" {text}:"
        print(f">>>{text} user time={u[0]:.2f}, sys time={u[1]:.2f}, mem={u[2] / 1024:.2f} MiB")
