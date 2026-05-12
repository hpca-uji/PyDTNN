"""
Utilities for profiling performance and memory usage in PyDTNN.
"""
import gc
import logging
import tempfile
import time
from pathlib import Path

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
    def __init__(self):
        """Initialize the profiler with an empty list of events."""
        self.events = []

    def start(self):
        """Start the profiling session."""
        raise NotImplementedError()

    def stop(self):
        """Stop the profiling session and record the result."""
        raise NotImplementedError()

    def __enter__(self):
        """Enter the context manager."""
        self.start()
        return self

    def __exit__(self, cls, exc, tb):
        """Exit the context manager."""
        self.stop()


class TimeProfiler(Profiler):
    """Profiler for measuring execution time."""
    def start(self):
        """Record the start time."""
        self._start_time = time.perf_counter()

    def stop(self):
        """Calculate elapsed time and append to events."""
        end_time = time.perf_counter()
        delta_time = end_time - self._start_time
        self.events.append(delta_time)


class MemoryProfiler(Profiler):
    """Profiler for measuring peak memory usage."""
    def start(self):
        """Initialize memory tracking."""
        self._tmp = tempfile.mktemp()
        self._tracer = memray.Tracker(self._tmp, native_traces=True, follow_fork=True)
        gc.collect()
        self._tracer.__enter__()

    def stop(self):
        """Stop tracking, compute peak memory, and clean up temporary files."""
        self._tracer.__exit__(None, None, None)
        stats = memray_statistics(self._tmp)
        Path(self._tmp).unlink()
        delta_memory = stats.metadata.peak_memory
        self.events.append(delta_memory)