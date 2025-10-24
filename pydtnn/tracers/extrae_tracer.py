import ctypes
import os
from importlib import import_module

from pydtnn.tracers.tracer import Tracer


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pydtnn.model import Model


class ExtraeTracer(Tracer):
    """
    ExtraTracer
    """

    def __init__(self, tracing: bool):
        super().__init__(tracing)
        self.pyextrae = None  # Declared here, will be initialized on enable_tracing()

    def enable_tracing(self):
        super().enable_tracing()
        self.pyextrae = import_module('pyextrae.common.extrae')

    def _define_event_types(self, model: "Model"):
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

    def _emit_event(self, evt_type: int, val: int, stream=None):
        """This method will be called only if tracing is enabled"""
        self.pyextrae.eventandcounters(evt_type, val)

    def _emit_nevent(self, evt: int, val: int, stream=None):
        """This method will be called only if tracing is enabled"""
        self.pyextrae.neventandcounters(evt, val)
