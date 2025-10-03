"""
Python interface to the PMLib library
"""
import ctypes
import ctypes.util
import functools

import numpy as np

from pydtnn.utils import load_library

_SERVER_IP_LEN = 16
_MAX_TIMING = 10
_LINE_SETSIZE = 16
_N_LINE_BITS = 128


class PMLibServer(ctypes.Structure):
    _fields_ = [("server_ip", ctypes.c_char * _SERVER_IP_LEN),
                ("port", ctypes.c_int)]


class PMLibLines(ctypes.Structure):
    _fields_ = [("__bits", ctypes.c_char * _LINE_SETSIZE)]


class PMLibMeasures(ctypes.Structure):
    _fields_ = [("watts_size", ctypes.c_int),
                ("watts_sets_size", ctypes.c_int),
                ("watts_sets", ctypes.POINTER(ctypes.c_int)),
                ("watts", ctypes.POINTER(ctypes.c_double)),
                ("lines_len", ctypes.c_int)]


class PMLibMeasuresWT(ctypes.Structure):
    _fields_ = [("next_timing", ctypes.c_int),
                ("timing", ctypes.POINTER(ctypes.c_double)),
                ("energy", PMLibMeasures)]


class PMLibCounter(ctypes.Structure):
    _fields_ = [("sock", ctypes.c_int),
                ("aggregate", ctypes.c_int),
                ("lines", PMLibLines),
                ("num_lines", ctypes.c_int),
                ("interval", ctypes.c_int),
                ("measures", ctypes.POINTER(PMLibMeasuresWT))]


class PMLibException(Exception):

    def __init__(self, error):
        self.error = error

    def __str__(self):
        return f'{self.error}'


def check_pmlib_returned_status(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        status = func(*args, **kwargs)
        if status != 0:
            raise PMLibException(f"Call to '{func.__name__}' failed!") from None
        return None

    return wrapper


class PMLib:
    _pmlib = None

    def __init__(self, server_ip, port, verbose=False):
        if self._pmlib is None:
            self._pmlib = load_library("pmlib")
        self.verbose = verbose
        # ----------------
        # Helper functions
        # ----------------
        # int pm_set_server( char *ip, int port, server_t *pm_server);
        self._pmlib.pm_set_server.restype = int
        self._pmlib.pm_set_server.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(PMLibServer)]
        # int pm_set_lines( char *lines_string ,line_t *lines );
        self._pmlib.pm_set_lines.restype = int
        self._pmlib.pm_set_lines.argtypes = [ctypes.c_char_p, ctypes.POINTER(PMLibLines)]
        # int pm_create_counter(char *pm_id, line_t lines, int aggregate, int interval, server_t pm_server,
        #                       counter_t *pm_counter);
        self._pmlib.pm_create_counter.restype = int
        self._pmlib.pm_create_counter.argtypes = [ctypes.c_char_p, PMLibLines, ctypes.c_int, ctypes.c_int,
                                                  PMLibServer, ctypes.POINTER(PMLibCounter)]
        # int pm_start_counter( counter_t *pm_counter );
        self._pmlib.pm_start_counter.restype = int
        self._pmlib.pm_start_counter.argtypes = [ctypes.POINTER(PMLibCounter)]
        # int pm_stop_counter( counter_t *pm_counter );
        self._pmlib.pm_stop_counter.restype = int
        self._pmlib.pm_stop_counter.argtypes = [ctypes.POINTER(PMLibCounter)]
        # int pm_get_counter_data( counter_t *pm_counter );
        self._pmlib.pm_get_counter_data.restype = int
        self._pmlib.pm_get_counter_data.argtypes = [ctypes.POINTER(PMLibCounter)]
        # int pm_print_data_text(char *file_name,  counter_t pm_counter, line_t lines, int set);
        self._pmlib.pm_print_data_text.restype = int
        self._pmlib.pm_print_data_text.argtypes = [ctypes.c_char_p, PMLibCounter, PMLibLines, ctypes.c_int]
        # int pm_finalize_counter( counter_t *pm_counter );
        self._pmlib.pm_finalize_counter.restype = int
        self._pmlib.pm_finalize_counter.argtypes = [ctypes.POINTER(PMLibCounter)]
        # -----------------------
        # Connect with the server
        # -----------------------
        self.server = PMLibServer()
        self.lines = PMLibLines()
        self.counter = PMLibCounter()
        self.set_server(server_ip, port)
        self.create_lines("0-15")
        # -----------------------
        # Class properties that will be initialized later
        # -----------------------
        self.counter_start_time = None
        self.counter_end_time = None
        self.period = None
        self.len_lines = None
        self.len_samples = None
        self.times = None
        self.watts = None

    def info(self, msg):
        if self.verbose is True:
            print("[PMLib]:", msg)

    @check_pmlib_returned_status
    def set_server(self, server_ip, port):
        self.info("Setting server...")
        return self._pmlib.pm_set_server(server_ip.encode('utf-8'), port, ctypes.byref(self.server))

    @check_pmlib_returned_status
    def create_lines(self, lines_string):
        self.info("Setting lines...")
        return self._pmlib.pm_set_lines(lines_string.encode('utf-8'), ctypes.byref(self.lines))

    @check_pmlib_returned_status
    def create_counter(self, counter_string, aggregate=0, interval=0):
        self.info("Creating counter...")
        return self._pmlib.pm_create_counter(counter_string.encode('utf-8'), self.lines, aggregate, interval,
                                             self.server, ctypes.byref(self.counter))

    @check_pmlib_returned_status
    def start_counter(self):
        self.info("Starting counter...")
        return self._pmlib.pm_start_counter(ctypes.byref(self.counter))

    @check_pmlib_returned_status
    def stop_counter(self):
        self.info("Stopping counter...")
        return self._pmlib.pm_stop_counter(ctypes.byref(self.counter))

    @check_pmlib_returned_status
    def _get_counter_data(self):
        self.info("Getting counter data...")
        return self._pmlib.pm_get_counter_data(ctypes.byref(self.counter))

    @check_pmlib_returned_status
    def print_data_text(self, output_string, set_value):
        self.info(f"Writing data to '{output_string}' file...")
        return self._pmlib.pm_print_data_text(output_string.encode('utf-8'), self.counter, self.lines, set_value)

    @check_pmlib_returned_status
    def finalize_counter(self):
        self.info("Finalizing counter...")
        return self._pmlib.pm_finalize_counter(ctypes.byref(self.counter))

    def get_counter_data(self):
        self._get_counter_data()
        self.counter_start_time, self.counter_end_time = np.ctypeslib.as_array(
            (ctypes.c_double * 2).from_address(ctypes.addressof(self.counter.measures.contents.timing.contents)))
        self.len_lines = 1 if self.counter.aggregate == 1 else self.counter.measures.contents.energy.lines_len
        self.len_samples = self.counter.measures.contents.energy.watts_size // self.len_lines
        self.period = (self.counter_end_time - self.counter_start_time) / (self.len_samples - 1)
        self.times = np.array([self.counter_start_time + x * self.period for x in range(self.len_samples)])
        self.watts = np.ctypeslib \
            .as_array((ctypes.c_double * self.len_samples * self.len_lines).from_address(
                ctypes.addressof(self.counter.measures.contents.energy.watts.contents))) \
            .reshape((self.len_lines, self.len_samples))
        if self.counter.aggregate == 0:
            _sum = np.sum(self.watts, axis=0).reshape(1, -1)
            self.watts = np.concatenate((_sum, self.watts))
            self.len_lines += 1

    def _next_sample_from_start(self, start_time):
        return min(self.len_samples - 1, int((start_time - self.times[0]) / self.period) + 1)

    def _previous_sample_from_end(self, end_time):
        return max(0, int(np.ceil((end_time - self.times[0]) / self.period)) - 1)

    def get_number_of_intermediate_samples(self, start_time, end_time):
        # Next and previous samples from start_time and end_time, respectively
        next_sample_from_start = self._next_sample_from_start(start_time)
        previous_sample_from_end = self._previous_sample_from_end(end_time)
        return max(0, previous_sample_from_end + 1 - next_sample_from_start)

    def get_joules(self, start_time, end_time, debug=False):
        # Check boundaries
        if start_time >= end_time:
            raise ValueError("End time must be greater than start time")
        if start_time < self.times[0]:
            raise ValueError("Given start time is lesser than the counter first time")
        if end_time > self.times[-1]:
            raise ValueError("Given end time is greater than the counter last time")
        # Next and previous samples from start_time and end_time, respectively
        next_sample_from_start = self._next_sample_from_start(start_time)
        previous_sample_from_end = self._previous_sample_from_end(end_time)
        if debug:
            print(f">> {next_sample_from_start=}")
            print(f">> {previous_sample_from_end=}")
        # Interpolate watts for start and end time
        watts_on_start_time, watts_on_end_time = [], []
        for watts in self.watts:
            a, b = np.interp([start_time, end_time], self.times, watts)
            watts_on_start_time.append(a)
            watts_on_end_time.append(b)
        # Promote watts_on_start_time and watts_on_end_time to np.arrays
        watts_on_start_time = np.array(watts_on_start_time)
        watts_on_end_time = np.array(watts_on_end_time)
        if debug:
            print(f">> {watts_on_start_time[0]=} ({self.watts[0, next_sample_from_start]=})")
            print(f">> {watts_on_end_time[0]=} ({self.watts[0, previous_sample_from_end]=})")
        # Integrate the energy
        if next_sample_from_start > previous_sample_from_end:
            # Integrate the energy between the two interpolated samples
            joules = ((watts_on_start_time + watts_on_end_time) / 2) * (end_time - start_time)
        else:
            joules = 0
            # Integrate the energy between start_time and times[next_sample_from_start]
            elapsed_time = self.times[next_sample_from_start] - start_time
            if elapsed_time > 0:
                joules += ((watts_on_start_time + self.watts[:, next_sample_from_start]) / 2) * elapsed_time
            # Integrate the energy between times[previous_sample_from_end] and end_time
            elapsed_time = end_time - self.times[previous_sample_from_end]
            if elapsed_time > 0:
                joules += ((self.watts[:, previous_sample_from_end] + watts_on_end_time) / 2) * elapsed_time
            # Integrate the energy between next_sample_from_start and previous_sample_from_end
            elapsed_time = self.times[previous_sample_from_end] - self.times[next_sample_from_start]
            if elapsed_time > 0:
                joules += np.mean(self.watts[:, next_sample_from_start:previous_sample_from_end + 1], axis=1) \
                    * elapsed_time
        return joules
