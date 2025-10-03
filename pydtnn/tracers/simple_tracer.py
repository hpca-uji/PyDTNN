import atexit
from collections import defaultdict
from timeit import default_timer as timer

from pydtnn.tracers.tracer import Tracer

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pympi.MPI import Comm as MPI_COMM
else:
    from types import ModuleType
    MPI_COMM = ModuleType


class SimpleTracer(Tracer):
    """
    SimpleTracer
    """

    def __init__(self, tracing: bool, output_filename: str, comm: MPI_COMM | None):
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

    def _emit_event(self, evt_type_val: int, evt_val: int, stream=None):
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

    def _emit_nevent(self, evt_type_val_list: list[int], evt_val_list: list[int], stream=None):
        """This method will be called only if tracing is enabled"""
        zipped_list = list(zip(evt_type_val_list, evt_val_list))
        if evt_val_list[0] == 0:
            zipped_list = reversed(zipped_list)
        for evt_type_val, evt_val in zipped_list:
            self.emit_event(evt_type_val, evt_val)

    def _output_header(self) -> str:
        return "Event type;Event value;Event name;Calls;Total time;Median of times"

    def _output_row(self, event_type_value: int, event_value: int) -> str:
        event_type = self.event_types[event_type_value]
        event_type_name = event_type.name
        _calls, _times = self.events[event_type_value][event_value]
        _times.sort()
        total_time = sum(_times)
        mean_of_times = _times[len(_times) // 2]
        return f"{event_type_name};{event_value};{event_type[event_value]};{_calls};{total_time};{mean_of_times}"

    def _write_output(self):
        """This method will be called at exit only if tracing has been enabled at any time"""
        if self.rank == 0:
            if len(self.pending_events):
                print("Warning: finishing simple tracer while there are pending events to be marked as finished.")
            print(f"Writing simple tracer output to '{self.output_filename}'...")
            with open(self.output_filename, 'w') as f:
                f.write(self._output_header() + "\n")
                for event_type_value, events in self.events.items():
                    for event_value in events.keys():
                        f.write(self._output_row(event_type_value, event_value) + "\n")
