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
import time
from collections import defaultdict

from . import SimpleTracer
from ..power_meters import PMLib


class SimpleTracerPMLib(SimpleTracer):
    """
    SimpleTracerPMLib
    """

    def __init__(self, tracing, output_filename, comm, pmlib_server_ip, pmlib_port, pmlib_device):
        super().__init__(tracing, output_filename, comm)
        if self.rank == 0:
            self.pmlib = PMLib(pmlib_server_ip, pmlib_port, verbose=True)
            self.pmlib_device = pmlib_device
        self.times = defaultdict(lambda: defaultdict(lambda: []))
        self.pending_times = []

    def enable_tracing(self):
        super().enable_tracing()
        # Start counter
        if self.rank == 0:
            self.pmlib.create_counter(self.pmlib_device)
            self.pmlib.start_counter()

    def _emit_event(self, evt_type_val, evt_val, stream=None):
        """This method will be called only if tracing is enabled"""
        super()._emit_event(evt_type_val, evt_val, stream)
        if evt_val != 0:
            self.pending_times.append((evt_type_val, evt_val, time.time()))
        else:
            end_time = time.time()
            _evt_type_val, _evt_val, start_time = self.pending_times.pop()
            self.times[_evt_type_val][_evt_val].append((start_time, end_time))

    def _output_header(self):
        output = super()._output_header()
        return output + ";Joules;Mean of intermediate power samples"

    def _output_row(self, event_type_value, event_value):
        output = super()._output_row(event_type_value, event_value)
        joules = 0
        intermediate_samples = 0
        for start_time, end_time in self.times[event_type_value][event_value]:
            joules += self.pmlib.get_joules(start_time, end_time)
            intermediate_samples += self.pmlib.get_number_of_intermediate_samples(start_time, end_time)
        if len(self.times[event_type_value][event_value]) > 0:
            intermediate_samples = intermediate_samples / len(self.times[event_type_value][event_value])
        return output + f";{joules};{intermediate_samples}"

    def _write_output(self):
        """This method will be called at exit only if tracing has been enabled at any time"""
        if self.rank == 0:
            self.pmlib.stop_counter()
            self.pmlib.get_counter_data()
            super()._write_output()
            self.pmlib.finalize_counter()