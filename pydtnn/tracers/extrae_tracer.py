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

import ctypes
import os
from importlib import import_module

from .tracer import Tracer


class ExtraeTracer(Tracer):
    """
    ExtraTracer
    """

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

    def _emit_event(self, evt_type, val, stream=None):
        """This method will be called only if tracing is enabled"""
        self.pyextrae.eventandcounters(evt_type, val)

    def _emit_nevent(self, evt, val, stream=None):
        """This method will be called only if tracing is enabled"""
        self.pyextrae.neventandcounters(evt, val)
