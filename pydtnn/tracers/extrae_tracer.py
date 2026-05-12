"""
Extrae tracer implementation for PyDTNN.
"""

from __future__ import annotations

import ctypes
import logging
import os
from importlib import import_module
from typing import TYPE_CHECKING

from pydtnn.tracers.tracer import Tracer

__all__ = ("ExtraeTracer",)

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from pydtnn.model import Model


class ExtraeTracer(Tracer):
    """
    Tracer implementation using the Extrae instrumentation library.
    """

    def __init__(self, tracing: bool):
        """
        Initialize the Extrae tracer.

        Args:
            tracing: Whether tracing is enabled.
        """
        super().__init__(tracing)
        self.pyextrae = None  # Declared here, will be initialized on enable_tracing()

    def enable_tracing(self):
        """
        Enable tracing and load the pyextrae module.
        """
        super().enable_tracing()
        self.pyextrae = import_module("pyextrae.common.extrae")

    def _define_event_types(self, model: Model):
        """
        Define event types in Extrae.

        Args:
            model: The model instance to extract event types from.
        """
        super()._define_event_types(model)
        for event_type_value, event_type in self.event_types.items():
            description = event_type.name
            nvalues = len(event_type)
            values = (ctypes.c_ulonglong * nvalues)()
            descriptions = (ctypes.c_char_p * nvalues)()
            for i, description in event_type.items():
                values[i] = i
                descriptions[i] = description
            assert self.pyextrae
            self.pyextrae.Extrae[os.getpid()].Extrae_define_event_type(
                ctypes.pointer(ctypes.c_uint(event_type_value)),
                ctypes.c_char_p(description.encode("utf-8")),
                ctypes.pointer(ctypes.c_uint(nvalues)),
                ctypes.pointer(values),
                ctypes.pointer(descriptions),
            )

    def _emit_event(self, evt_type: int, val: int, stream=None):
        """
        Emit a single event to Extrae.

        Args:
            evt_type: The event type identifier.
            val: The event value.
            stream: Optional stream identifier.
        """
        assert self.pyextrae
        self.pyextrae.eventandcounters(evt_type, val)

    def _emit_nevent(self, evt: int, val: int, stream=None):
        """
        Emit a nested event to Extrae.

        Args:
            evt: The event type identifier.
            val: The event value.
            stream: Optional stream identifier.
        """
        assert self.pyextrae
        self.pyextrae.neventandcounters(evt, val)
