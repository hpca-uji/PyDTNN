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

import atexit
from collections import defaultdict
from timeit import default_timer as timer

from .tracer import Tracer


class SimpleTracer(Tracer):
    """
    SimpleTracer
    """

    def __init__(self, tracing, output_filename, comm):
        super().__init__(tracing)
        self.output_filename = output_filename
        self.rank = 0
        if comm is not None:
            self.rank = comm.Get_rank()
        self.events = defaultdict(lambda: defaultdict(lambda: [0, []]))
        self.pending_events = []

    def enable_tracing(self):
        super().enable_tracing()
        # If tracing is enabled at least once, register self.write_output to be executed at exit
        atexit.register(self._write_output)

    def _emit_event(self, evt_type_val, evt_val, stream=None):
        """This method will be called only if tracing is enabled"""
        if evt_val != 0:
            self.pending_events.append((evt_type_val, evt_val, timer()))
        else:
            toc = timer()
            if len(self.pending_events) == 0:
                raise RuntimeError("Received an 'End' event but there are no pending events!")
            if self.pending_events[-1][0] != evt_type_val:
                raise RuntimeError("Received an 'End' event for a different event type than expected!")
            _evt_type_val, _evt_val, tic = self.pending_events.pop()
            self.events[_evt_type_val][_evt_val][0] += 1
            self.events[_evt_type_val][_evt_val][1].append(toc - tic)

    def _emit_nevent(self, evt_type_val_list, evt_val_list, stream=None):
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
                f.write("Event type;Event value;Event name;Calls;Total time;Median of times\n")
                for event_type_key, events in self.events.items():
                    event_type = self.event_types[event_type_key]
                    event_type_name = event_type.name
                    for value, (_calls, _times) in events.items():
                        _times.sort()
                        total_time = sum(_times)
                        mean_of_times = _times[len(_times) // 2]
                        f.write(f"{event_type_name};{value};{event_type[value]};{_calls};{total_time};{mean_of_times}\n")
