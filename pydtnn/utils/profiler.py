import gc
import time
import memray
import tempfile
from pathlib import Path
from memray._memray import compute_statistics as memray_statistics


class Profiler:
    def __init__(self):
        self.events = []

    def start(self):
        raise NotImplementedError()

    def stop(self):
        raise NotImplementedError()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, cls, exc, tb):
        self.stop()


class TimeProfiler(Profiler):
    def start(self):
        self._start_time = time.perf_counter()

    def stop(self):
        end_time = time.perf_counter()
        delta_time = end_time - self._start_time
        self.events.append(delta_time)


class MemoryProfiler(Profiler):
    def start(self):
        self._tmp = tempfile.mktemp()
        self._tracer = memray.Tracker(self._tmp, native_traces=True, follow_fork=True)
        gc.collect()
        self._tracer.__enter__()

    def stop(self):
        self._tracer.__exit__(None, None, None)
        stats = memray_statistics(self._tmp)
        Path(self._tmp).unlink()
        delta_memory = stats.metadata.peak_memory
        self.events.append(delta_memory)
