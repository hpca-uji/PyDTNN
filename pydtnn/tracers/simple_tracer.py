"""Simple tracer implementation for PyDTNN."""

import atexit
import logging
from collections import defaultdict
from pathlib import Path
from timeit import default_timer as timer
from types import ModuleType
from typing import TYPE_CHECKING

from pydtnn import utils
from pydtnn.tracers.tracer import StreamType, Tracer

__all__ = ("SimpleTracer",)

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from pympi.MPI import Comm  # type: ignore
else:
    from types import ModuleType

    Comm = ModuleType


class SimpleTracer(Tracer):
    """A basic tracer implementation that records event durations and writes them to a CSV file."""

    def __init__(self, tracing: bool, output_filename: str, comm: Comm | None) -> None:
        """
        Initialize the SimpleTracer.

        Args:
            tracing: Whether tracing is enabled.
            output_filename: Path to the output file.
            comm: MPI communicator for rank-aware output.
        """
        super().__init__(tracing)
        self.output_filename = output_filename
        self.rank = 0
        if comm is not None:
            self.rank = comm.Get_rank()
        self.events = defaultdict(
            lambda: defaultdict(lambda: [0, []])
        )  # TODO: use tuple or structure
        self.pending_events = []

    def enable_tracing(self) -> None:
        """Enable tracing and register the output writer to run at exit."""
        super().enable_tracing()
        # If tracing is enabled at least once, register self.write_output to be executed at exit
        atexit.register(self._write_output)

    def _emit_event(self, evt_type_val: int, evt_val: int, stream: StreamType | None = None) -> None:
        """
        Record the start or end of an event.

        Args:
            evt_type_val: The integer identifier for the event type.
            evt_val: The integer identifier for the specific event value.
            stream: Optional stream context.
        """
        """This method will be called only if tracing is enabled"""
        if evt_val != 0:
            self.pending_events.append((evt_type_val, evt_val, timer()))
        else:
            toc = timer()
            if len(self.pending_events) == 0:
                raise RuntimeError("Received an 'End' event but there are no pending events!")
            if self.pending_events[-1][0] != evt_type_val:
                raise RuntimeError(
                    "Received an 'End' event for a different event type than expected!"
                )
            _evt_type_val, _evt_val, tic = self.pending_events.pop()
            self.events[_evt_type_val][_evt_val][0] += 1  # type: ignore
            self.events[_evt_type_val][_evt_val][1].append(toc - tic)  # type: ignore

    def _emit_nevent(self, evt_type_val_list: list[int], evt_val_list: list[int], stream: StreamType | None = None) -> None:
        """
        Record multiple events simultaneously.

        Args:
            evt_type_val_list: List of event type identifiers.
            evt_val_list: List of event value identifiers.
            stream: Optional stream context.
        """
        """This method will be called only if tracing is enabled"""
        zipped_list = list(zip(evt_type_val_list, evt_val_list))
        if evt_val_list[0] == 0:
            zipped_list = reversed(zipped_list)
        for evt_type_val, evt_val in zipped_list:
            self.emit_event(evt_type_val, evt_val)

    def _output_header(self) -> str:
        """Return the CSV header string for the output file."""

        return "Event type,Event value,Event name,Calls,Total time,Median of times"
        # return "Event type;Event value;Event name;Calls;Total time;Median of times"

    def _output_row(self, event_type_value: int, event_value: int) -> str:
        """
        Format a single event row for the output file.

        Args:
            event_type_value: The event type identifier.
            event_value: The event value identifier.
        """
        event_type = self.event_types[event_type_value]
        event_type_name = event_type.name
        _calls, _times = self.events[event_type_value][event_value]
        assert isinstance(_times, list)
        _times.sort()
        total_time = sum(_times)
        mean_of_times = _times[len(_times) // 2]
        return f"{event_type_name},{event_value},{event_type[event_value]},{_calls},{total_time},{
            mean_of_times
        }"
        # return
        # f"{event_type_name};{event_value};{event_type[event_value]};{_calls};{total_time};{mean_of_times}"

    def _write_output(self) -> None:
        """
        Write the collected trace data to the output file.
        This method will be called at exit only if tracing has been enabled at any time
        """
        output_filename = utils.string_substitute(self.output_filename, rank=self.rank)  # type: ignore (It's fine)
        if output_filename != self.output_filename or self.rank == 0:
            if len(self.pending_events):
                logger.warning(
                    "Warning: finishing simple tracer while there are pending events to be marked"
                    " as finished."
                )
            path = Path(output_filename).resolve()
            with open(output_filename, "w") as f:
                f.write(self._output_header() + "\n")
                for event_type_value, events in self.events.items():
                    for event_value in events.keys():
                        f.write(self._output_row(event_type_value, event_value) + "\n")
            logger.info(f"Dumped tracer details to: {path}")
