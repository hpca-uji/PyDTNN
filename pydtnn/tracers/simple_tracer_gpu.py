"""GPU-accelerated implementation of the SimpleTracer using PyCUDA."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import pycuda.driver as drv  # type: ignore

from pydtnn.tracers.simple_tracer import SimpleTracer
from pydtnn.tracers.tracer import StreamType

type drvEvent = Any  # drv.Event()

__all__ = ("SimpleTracerPycuda",)

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from pympi.MPI import Comm  # type: ignore


class SimpleTracerPycuda(SimpleTracer):
    """A tracer implementation that records GPU execution times using PyCUDA events."""

    def __init__(self, tracing: bool, output_filename: str, comm: Comm | None) -> None:
        """
        Initialize the PyCUDA tracer.

        Args:
            tracing: Whether tracing is enabled.
            output_filename: Path to the output file for trace data.
            comm: MPI communicator for distributed tracing.
        """
        super().__init__(tracing, output_filename, comm)
        self.event_vars = []
        # Attributes that will be initialized later
        self.stream: drv.Stream = None

    def _get_start_end_event(self) -> tuple:
        """
        Retrieve a pair of PyCUDA events for timing.

        Returns:
            A tuple containing (start_event, end_event).
        """
        if len(self.event_vars) == 0:
            self.event_vars.append((drv.Event(), drv.Event()))
        return self.event_vars.pop()

    def _release_start_end_event(self, start: drvEvent, end: drvEvent) -> None:
        """Return a pair of PyCUDA events to the pool for reuse."""
        self.event_vars.append((start, end))

    def _emit_event(
        self, evt_type_val: int, evt_val: int, stream: StreamType | None = None
    ) -> None:
        """
        Record a single GPU event duration.

        Args:
            evt_type_val: The category identifier of the event.
            evt_val: The specific event identifier (0 for end, non-zero for start).
            stream: The PyCUDA stream to record on.
        """
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
                raise RuntimeError(
                    "Received an 'End' event for a different event type than expected!"
                )
            _evt_type_val, _evt_val, start, end = self.pending_events.pop()
            end.record(stream=stream)
            end.synchronize()
            evt_time = start.time_till(end) * 1e-3
            self._release_start_end_event(start, end)
            previous_calls, previous_time = self.events[_evt_type_val][_evt_val]
            self.events[_evt_type_val][_evt_val] = [
                previous_calls + 1,  # type: ignore (previous_calls is an int)
                previous_time + evt_time,
            ]  # type: ignore

    def _emit_nevent(
        self, evt_type_val_list: list, evt_val_list: list, stream: StreamType | None = None
    ) -> None:
        """
        Record multiple GPU events.

        Args:
            evt_type_val_list: List of event category identifiers.
            evt_val_list: List of event identifiers.
            stream: The PyCUDA stream to record on.
        """
        zipped_list = list(zip(evt_type_val_list, evt_val_list))
        if evt_val_list[0] == 0:
            zipped_list = reversed(zipped_list)
        for evt_type_val, evt_val in zipped_list:
            self.emit_event(evt_type_val, evt_val, stream)

    def set_stream(self, stream: StreamType) -> None:
        """Set the default PyCUDA stream for event recording."""
        self.stream = stream
