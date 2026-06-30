"""Python interface to the PMLib library"""

import ctypes
import functools
import logging
from typing import Callable

import numpy as np

from pydtnn.utils import load_library

__all__ = (
    "PMLib",
    "PMLibCounter",
    "PMLibException",
    "PMLibLines",
    "PMLibMeasures",
    "PMLibMeasuresWT",
    "PMLibServer",
    "check_pmlib_returned_status",
)

logger = logging.getLogger(__name__)


_SERVER_IP_LEN = 16
_MAX_TIMING = 10
_LINE_SETSIZE = 16
_N_LINE_BITS = 128


class PMLibServer(ctypes.Structure):
    """Structure representing a PMLib server connection configuration."""

    _fields_ = [("server_ip", ctypes.c_char * _SERVER_IP_LEN), ("port", ctypes.c_int)]


class PMLibLines(ctypes.Structure):
    """Structure representing a set of lines for power measurement."""

    _fields_ = [("__bits", ctypes.c_char * _LINE_SETSIZE)]


class PMLibMeasures(ctypes.Structure):
    """Structure containing power measurement data."""

    _fields_ = [
        ("watts_size", ctypes.c_int),
        ("watts_sets_size", ctypes.c_int),
        ("watts_sets", ctypes.POINTER(ctypes.c_int)),
        ("watts", ctypes.POINTER(ctypes.c_double)),
        ("lines_len", ctypes.c_int),
    ]


class PMLibMeasuresWT(ctypes.Structure):
    """Structure containing power measurement data with timing information."""

    _fields_ = [
        ("next_timing", ctypes.c_int),
        ("timing", ctypes.POINTER(ctypes.c_double)),
        ("energy", PMLibMeasures),
    ]


class PMLibCounter(ctypes.Structure):
    """Structure representing a power measurement counter."""

    _fields_ = [
        ("sock", ctypes.c_int),
        ("aggregate", ctypes.c_int),
        ("lines", PMLibLines),
        ("num_lines", ctypes.c_int),
        ("interval", ctypes.c_int),
        ("measures", ctypes.POINTER(PMLibMeasuresWT)),
    ]


class PMLibException(Exception):
    """Exception raised for errors occurring within the PMLib interface."""

    def __init__(self, error: str) -> None:
        """Initialize the exception with an error message."""
        self.error = error

    def __str__(self) -> str:
        """Return the string representation of the exception."""
        return f"{self.error}"


def check_pmlib_returned_status(func: Callable) -> Callable:
    """Decorator to check the return status of PMLib C functions.

    This decorator wraps a function that calls a PMLib C function.
    It checks if the return status is non-zero, indicating an error.
    If an error occurs, it raises a PMLibException.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        """Wrapper function to execute the decorated function and check its status."""
        status = func(*args, **kwargs)
        if status != 0:
            raise PMLibException(f"Call to '{func.__name__}' failed!") from None
        return None

    return wrapper


class PMLib:
    """Interface class for interacting with the PMLib library.

    This class provides a Python wrapper around the PMLib C library,
    enabling power measurement operations such as setting up servers,
    defining measurement lines, creating and managing counters, and
    retrieving measurement data.
    """

    _pmlib = None

    def __init__(self, server_ip: str, port: int, verbose: bool = False) -> None:
        """Initialize the PMLib interface with server details.

        Loads the PMLib shared library and sets up the connection parameters
        for the power measurement server. It also initializes internal
        structures and helper functions for interacting with the C library.

        Args:
            server_ip (str): The IP address of the power measurement server.
            port (int): The port number of the power measurement server.
            verbose (bool, optional): If True, enables verbose logging. Defaults to False.
        """
        if self._pmlib is None:
            self._pmlib = load_library("pmlib")
        self.verbose = verbose
        # Helper functions
        # int pm_set_server( char *ip, int port, server_t *pm_server);
        self._pmlib.pm_set_server.restype = int
        self._pmlib.pm_set_server.argtypes = [
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.POINTER(PMLibServer),
        ]
        # int pm_set_lines( char *lines_string ,line_t *lines );
        self._pmlib.pm_set_lines.restype = int
        self._pmlib.pm_set_lines.argtypes = [ctypes.c_char_p, ctypes.POINTER(PMLibLines)]
        # int pm_create_counter(char *pm_id, line_t lines, int aggregate, int interval, server_t pm_server,
        #                       counter_t *pm_counter);
        self._pmlib.pm_create_counter.restype = int
        self._pmlib.pm_create_counter.argtypes = [
            ctypes.c_char_p,
            PMLibLines,
            ctypes.c_int,
            ctypes.c_int,
            PMLibServer,
            ctypes.POINTER(PMLibCounter),
        ]
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
        self._pmlib.pm_print_data_text.argtypes = [
            ctypes.c_char_p,
            PMLibCounter,
            PMLibLines,
            ctypes.c_int,
        ]
        # int pm_finalize_counter( counter_t *pm_counter );
        self._pmlib.pm_finalize_counter.restype = int
        self._pmlib.pm_finalize_counter.argtypes = [ctypes.POINTER(PMLibCounter)]
        # Connect with the server
        self.server = PMLibServer()
        self.lines = PMLibLines()
        self.counter = PMLibCounter()
        self.set_server(server_ip, port)
        self.create_lines("0-15")
        # Class properties that will be initialized later
        self.counter_start_time: float = None  # type: ignore (It will be initalized later)
        self.counter_end_time: float = None  # type: ignore (It will be initalized later)
        self.period: float = None  # type: ignore (It will be initalized later)
        self.len_lines: int = None  # type: ignore (It will be initalized later)
        self.len_samples: int = None  # type: ignore (It will be initalized later)
        self.times: np.ndarray = None  # type: ignore (It will be initalized later)
        self.watts = None

    def info(self, msg: str) -> None:
        """Log informational messages if verbose mode is enabled.

        Args:
            msg (str): The message to log.
        """
        if self.verbose is True:
            logger.info("[PMLib]:", msg)

    @check_pmlib_returned_status
    def set_server(self, server_ip: str, port: int) -> None:
        """Configure the server connection.

        Sets the IP address and port for the PMLib server.

        Args:
            server_ip (str): The IP address of the server.
            port (int): The port number of the server.

        Returns:
            None: If the operation is successful.

        Raises:
            PMLibException: If the underlying C function call fails.
        """
        self.info("Setting server...")
        assert self._pmlib
        return self._pmlib.pm_set_server(server_ip.encode("utf-8"), port, ctypes.byref(self.server))

    @check_pmlib_returned_status
    def create_lines(self, lines_string: str) -> None:
        """Define the lines to be monitored.

        Parses a string representing the lines to be measured and configures
        the internal `PMLibLines` structure.

        Args:
            lines_string (str): A string specifying the lines, e.g., "0-15".

        Returns:
            None: If the operation is successful.

        Raises:
            PMLibException: If the underlying C function call fails.
        """
        self.info("Setting lines...")
        assert self._pmlib
        return self._pmlib.pm_set_lines(lines_string.encode("utf-8"), ctypes.byref(self.lines))

    @check_pmlib_returned_status
    def create_counter(self, counter_string: str, aggregate: int = 0, interval: int = 0) -> None:
        """Initialize a power measurement counter.

        Creates a counter object in the PMLib library, associated with a
        specific ID, lines, aggregation mode, and interval.

        Args:
            counter_string (str): A unique identifier string for the counter.
            aggregate (int, optional): Whether to aggregate measurements. Defaults to 0.
            interval (int, optional): The sampling interval in seconds. Defaults to 0.

        Returns:
            None: If the operation is successful.

        Raises:
            PMLibException: If the underlying C function call fails.
        """
        self.info("Creating counter...")
        assert self._pmlib
        return self._pmlib.pm_create_counter(
            counter_string.encode("utf-8"),
            self.lines,
            aggregate,
            interval,
            self.server,
            ctypes.byref(self.counter),
        )

    @check_pmlib_returned_status
    def start_counter(self) -> None:
        """Start the power measurement counter.

        Begins the data acquisition process for the configured counter.

        Returns:
            None: If the operation is successful.

        Raises:
            PMLibException: If the underlying C function call fails.
        """
        self.info("Starting counter...")
        assert self._pmlib
        return self._pmlib.pm_start_counter(ctypes.byref(self.counter))

    @check_pmlib_returned_status
    def stop_counter(self) -> None:
        """Stop the power measurement counter.

        Halts the data acquisition process for the configured counter.

        Returns:
            None: If the operation is successful.

        Raises:
            PMLibException: If the underlying C function call fails.
        """
        self.info("Stopping counter...")
        assert self._pmlib
        return self._pmlib.pm_stop_counter(ctypes.byref(self.counter))

    @check_pmlib_returned_status
    def _get_counter_data(self) -> None:
        """Internal method to fetch raw counter data from the library.

        Retrieves the latest measurement data from the PMLib counter.

        Returns:
            None: If the operation is successful.

        Raises:
            PMLibException: If the underlying C function call fails.
        """
        self.info("Getting counter data...")
        assert self._pmlib
        return self._pmlib.pm_get_counter_data(ctypes.byref(self.counter))

    @check_pmlib_returned_status
    def print_data_text(self, output_string: str, set_value: int) -> None:
        """Export counter data to a text file.

        Saves the current counter data to a specified text file.

        Args:
            output_string (str): The name of the output file.
            set_value (int): An integer representing the set to export.

        Returns:
            None: If the operation is successful.

        Raises:
            PMLibException: If the underlying C function call fails.
        """
        self.info(f"Writing data to '{output_string}' file...")
        assert self._pmlib
        return self._pmlib.pm_print_data_text(
            output_string.encode("utf-8"), self.counter, self.lines, set_value
        )

    @check_pmlib_returned_status
    def finalize_counter(self) -> None:
        """Finalize and clean up the counter resources.

        Releases resources associated with the PMLib counter.

        Returns:
            None: If the operation is successful.

        Raises:
            PMLibException: If the underlying C function call fails.
        """
        self.info("Finalizing counter...")
        assert self._pmlib
        return self._pmlib.pm_finalize_counter(ctypes.byref(self.counter))

    def get_counter_data(self) -> None:
        """Fetch, parse, and store counter data into numpy arrays.

        Retrieves raw data from the PMLib counter, parses it into time series
        of power measurements (Watts) and timestamps, and stores them as
        NumPy arrays in class attributes. It also calculates derived properties
        like the number of lines, samples, and the time period.
        """
        self._get_counter_data()
        self.counter_start_time, self.counter_end_time = np.ctypeslib.as_array(
            (ctypes.c_double * 2).from_address(
                ctypes.addressof(self.counter.measures.contents.timing.contents)
            )
        )
        self.len_lines = (
            1 if self.counter.aggregate == 1 else self.counter.measures.contents.energy.lines_len
        )
        self.len_samples = self.counter.measures.contents.energy.watts_size // self.len_lines
        self.period = (self.counter_end_time - self.counter_start_time) / (self.len_samples - 1)
        self.times = np.array(
            [self.counter_start_time + x * self.period for x in range(self.len_samples)]
        )
        self.watts = np.ctypeslib.as_array(
            (ctypes.c_double * self.len_samples * self.len_lines).from_address(
                ctypes.addressof(self.counter.measures.contents.energy.watts.contents)
            )
        ).reshape((self.len_lines, self.len_samples))
        if self.counter.aggregate == 0:
            _sum: np.ndarray = np.sum(self.watts, axis=0).reshape(1, -1)
            self.watts = np.concatenate((_sum, self.watts))
            self.len_lines += 1

    def _next_sample_from_start(self, start_time: float) -> int:
        """Calculate the index of the next sample relative to start_time.

        Determines the index of the sample in the `self.times` array that
        corresponds to or is immediately after the given `start_time`.

        Args:
            start_time (float): The reference start time.

        Returns:
            int: The index of the next sample.
        """
        assert self.len_samples
        assert self.times
        return min(self.len_samples - 1, int((start_time - self.times[0]) / self.period) + 1)

    def _previous_sample_from_end(self, end_time: float) -> int:
        """Calculate the index of the previous sample relative to end_time.

        Determines the index of the sample in the `self.times` array that
        corresponds to or is immediately before the given `end_time`.

        Args:
            end_time (float): The reference end time.

        Returns:
            int: The index of the previous sample.
        """
        assert self.times
        return max(0, int(np.ceil((end_time - self.times[0]) / self.period)) - 1)

    def get_number_of_intermediate_samples(self, start_time: float, end_time: float) -> int:
        """Return the count of samples between the given time range.

        Calculates the number of discrete time samples that fall strictly
        between the `start_time` and `end_time`.

        Args:
            start_time (float): The start of the time range.
            end_time (float): The end of the time range.

        Returns:
            int: The number of intermediate samples.
        """
        # Next and previous samples from start_time and end_time, respectively
        next_sample_from_start = self._next_sample_from_start(start_time)
        previous_sample_from_end = self._previous_sample_from_end(end_time)
        return max(0, previous_sample_from_end + 1 - next_sample_from_start)

    def get_joules(self, start_time: float, end_time: float, debug: bool = False) -> np.ndarray:
        """Calculate total energy in Joules for a specified time interval.

        Integrates the power measurements (Watts) over the given time interval
        (`start_time` to `end_time`) to compute the total energy consumed in Joules.
        It handles interpolation for partial samples at the interval boundaries.

        Args:
            start_time (float): The start of the time interval (in seconds).
            end_time (float): The end of the time interval (in seconds).
            debug (bool, optional): If True, enables debug logging. Defaults to False.

        Returns:
            np.ndarray: A NumPy array containing the total energy in Joules for
                        each line (including the aggregated sum if applicable).

        Raises:
            ValueError: If `start_time` is not less than `end_time`, or if the
                        given times are outside the range of recorded data.
        """
        # Check boundaries
        assert self.times
        if start_time >= end_time:
            raise ValueError("End time must be greater than start time")
        if start_time < self.times[0]:
            raise ValueError("Given start time is lesser than the counter first time")
        if end_time > self.times[-1]:
            raise ValueError("Given end time is greater than the counter last time")
        # Next and previous samples from start_time and end_time, respectively
        next_sample_from_start = self._next_sample_from_start(start_time)
        previous_sample_from_end = self._previous_sample_from_end(end_time)
        logger.debug(f">> {next_sample_from_start=}")
        logger.debug(f">> {previous_sample_from_end=}")
        # Interpolate watts for start and end time
        watts_on_start_time = list[np.float64]()
        watts_on_end_time = list[np.float64]()
        assert self.watts
        for watts in self.watts:
            a, b = np.interp([start_time, end_time], self.times, watts)
            watts_on_start_time.append(a)
            watts_on_end_time.append(b)
        # Promote watts_on_start_time and watts_on_end_time to np.arrays
        watts_on_start_time = np.array(watts_on_start_time)
        watts_on_end_time = np.array(watts_on_end_time)
        logger.debug(f">> {watts_on_start_time[0]=} ({self.watts[0, next_sample_from_start]=})")
        logger.debug(f">> {watts_on_end_time[0]=} ({self.watts[0, previous_sample_from_end]=})")
        # Integrate the energy
        if next_sample_from_start > previous_sample_from_end:
            # Integrate the energy between the two interpolated samples
            joules = ((watts_on_start_time + watts_on_end_time) / 2) * (end_time - start_time)
        else:
            joules: np.ndarray = 0  # type: ignore (It will change it's type later)
            # Integrate the energy between start_time and times[next_sample_from_start]
            elapsed_time: float = self.times[next_sample_from_start] - start_time
            if elapsed_time > 0:
                joules += (
                    (watts_on_start_time + self.watts[:, next_sample_from_start]) / 2
                ) * elapsed_time
            # Integrate the energy between times[previous_sample_from_end] and end_time
            elapsed_time = end_time - self.times[previous_sample_from_end]
            if elapsed_time > 0:
                joules += (
                    (self.watts[:, previous_sample_from_end] + watts_on_end_time) / 2
                ) * elapsed_time
            # Integrate the energy between next_sample_from_start and previous_sample_from_end
            elapsed_time = self.times[previous_sample_from_end] - self.times[next_sample_from_start]
            if elapsed_time > 0:
                joules += (
                    np.mean(
                        self.watts[:, next_sample_from_start: previous_sample_from_end + 1], axis=1
                    )
                    * elapsed_time
                )
        return joules
