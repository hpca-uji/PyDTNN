try:
    import pycuda.driver as drv  #type: ignore
except Exception as e:
    pass

from typing import TYPE_CHECKING

from pydtnn.tracers.simple_tracer import SimpleTracer

if TYPE_CHECKING:
    from pympi.MPI import Comm as MPI_COMM
else:
    from types import ModuleType
    MPI_COMM = ModuleType


class SimpleTracerGPU(SimpleTracer):
    """
    SimpleTracerGPU
    """

    def __init__(self, tracing: bool, output_filename: str, comm: MPI_COMM | None):
        super().__init__(tracing, output_filename, comm)
        self.event_vars = []
        # Attributes that will be initialized later
        self.stream: drv.Stream = None

    def _get_start_end_event(self) -> tuple:
        if len(self.event_vars) == 0:
            self.event_vars.append((drv.Event(), drv.Event()))
        return self.event_vars.pop()

    def _release_start_end_event(self, start, end):
        self.event_vars.append((start, end))

    def _emit_event(self, evt_type_val: int, evt_val: int, stream=None):
        """This method will be called only if tracing is enabled"""
        if stream is None:
            stream = self.stream
        if evt_val != 0:
            start, end = self._get_start_end_event()
            self.pending_events.append((evt_type_val, evt_val, start, end))
            start.record(stream=stream)
        else:
            if len(self.pending_events) == 0:
                raise RuntimeError("Received an 'End' event but there are no pending events!")
            if self.pending_events[-1][0] != evt_type_val:
                raise RuntimeError("Received an 'End' event for a different event type than expected!")
            _evt_type_val, _evt_val, start, end = self.pending_events.pop()
            end.record(stream=stream)
            end.synchronize()
            evt_time = start.time_till(end) * 1e-3
            self._release_start_end_event(start, end)
            previous_calls, previous_time = self.events[_evt_type_val][_evt_val]
            self.events[_evt_type_val][_evt_val] = [previous_calls + 1, previous_time + evt_time]

    def _emit_nevent(self, evt_type_val_list: list, evt_val_list: list, stream=None):
        """This method will be called only if tracing is enabled"""
        zipped_list = list(zip(evt_type_val_list, evt_val_list))
        if evt_val_list[0] == 0:
            zipped_list = reversed(zipped_list)
        for evt_type_val, evt_val in zipped_list:
            self.emit_event(evt_type_val, evt_val, stream)

    def set_stream(self, stream):
        self.stream = stream
