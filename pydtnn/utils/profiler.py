"""Utilities for profiling performance and memory usage in PyDTNN."""

import gc
import os
import time
import types
import logging
import tempfile
from pathlib import Path
from typing import Self

import memray  # type: ignore
from memray._memray import compute_statistics as memray_statistics  # type: ignore

__all__ = (
    "MemoryProfiler",
    "Profiler",
    "TimeProfiler",
)

logger = logging.getLogger(__name__)


class Profiler:
    """Base class for performance and resource profiling."""

    def __init__(self) -> None:
        """Initialize the profiler with an empty list of events."""
        self.events = []

    def start(self) -> None:
        """Start the profiling session."""
        raise NotImplementedError()

    def stop(self) -> None:
        """Stop the profiling session and record the result."""
        raise NotImplementedError()

    def __enter__(self) -> Self:
        """Enter the context manager."""
        self.start()
        return self

    def __exit__[T: Exception](self, cls: type[T], exc: T, tb: types.TracebackType) -> None:
        """Exit the context manager."""
        self.stop()


class TimeProfiler(Profiler):
    """Profiler for measuring execution time."""

    def start(self) -> None:
        """Record the start time."""
        self._start_time = time.perf_counter()

    def stop(self) -> None:
        """Calculate elapsed time and append to events."""
        end_time = time.perf_counter()
        delta_time = end_time - self._start_time
        self.events.append(delta_time)


class MemoryProfiler(Profiler):
    """Profiler for measuring peak memory usage."""

    def start(self) -> None:
        """Initialize memory tracking."""
        fd, self._tmp = tempfile.mkstemp()
        os.close(fd)
        Path(self._tmp).unlink()
        self._tracer = memray.Tracker(self._tmp, native_traces=True, follow_fork=True)
        gc.collect()
        self._tracer.__enter__()

    def stop(self) -> None:
        """Stop tracking, compute peak memory, and clean up temporary files."""
        self._tracer.__exit__(None, None, None)
        stats = memray_statistics(self._tmp)
        Path(self._tmp).unlink()
        delta_memory = stats.metadata.peak_memory
        self.events.append(delta_memory)
