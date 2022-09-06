"""
Python interface to the PMLib library
"""
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
#  with this program. If not, see <https://www.gnu.org/licenses/>.
#

import ctypes
import ctypes.util

from pydtnn.utils import load_library

_SERVER_IP_LEN = 16
_MAX_TIMING = 10
_LINE_SETSIZE = 16
_N_LINE_BITS = 128


class _PMLibServer(ctypes.Structure):
    _fields_ = [("server_ip", ctypes.c_char * _SERVER_IP_LEN),
                ("port", ctypes.c_int)]


class _PMLibLines(ctypes.Structure):
    _fields_ = [("__bits", ctypes.c_char * _LINE_SETSIZE)]


class _PMLibMeasures(ctypes.Structure):
    _fields_ = [("watts_size", ctypes.c_int),
                ("watts_sets_size", ctypes.c_int),
                ("watts_sets", ctypes.POINTER(ctypes.c_int)),
                ("watts", ctypes.POINTER(ctypes.c_double)),
                ("lines_len", ctypes.c_int)]


class _PMLibMeasuresWT(ctypes.Structure):
    _fields_ = [("next_timing", ctypes.c_int),
                ("timing", ctypes.POINTER(ctypes.c_double)),
                ("energy", _PMLibMeasures)]


class _PMLibCounter(ctypes.Structure):
    _fields_ = [("sock", ctypes.c_int),
                ("aggregate", ctypes.c_int),
                ("lines", _PMLibLines),
                ("num_lines", ctypes.c_int),
                ("interval", ctypes.c_int),
                ("measures", _PMLibMeasuresWT)]


class PMLibException(Exception):

    def __init__(self, error):
        self.error = error

    def __str__(self):
        return f'{self.error}'


class PMLib:
    _pmlib = None

    def __init__(self):
        if self._pmlib is None:
            self._pmlib = load_library("pmlib")

        # Helper functions

        # # int pm_set_server( char *ip, int port, server_t *pm_server);
        #self._pmlib.pm_set_server.restype = int
        #self._pmlib.pm_set_server.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(_PMLibServer)]

        # # int pm_set_lines( char *lines_string ,line_t *lines );
        # self._pmlib.pm_set_lines.restype = int
        # self._pmlib.pm_set_lines.argtypes = [ctypes.c_char_p, ctypes.POINTER(PMLibLines)]

        # # int pm_create_counter(char *pm_id, line_t lines, int aggregate, int interval, server_t pm_server,
        # #                       counter_t *pm_counter);
        # self._pmlib.pm_create_counter.restype = int
        # self._pmlib.pm_create_counter.argtypes = [ctypes.c_char_p, PMLibLines, ctypes.c_int, ctypes.c_int,
        #                                           PMLibServer, ctypes.POINTER(PMLibCounter)]

        # # int pm_start_counter( counter_t *pm_counter );
        # self._pmlib.pm_start_counter.restype = int
        # self._pmlib.pm_start_counter.argtypes = [ctypes.POINTER(PMLibCounter)]

        # # int pm_stop_counter( counter_t *pm_counter );
        # self._pmlib.pm_stop_counter.restype = int
        # self._pmlib.pm_stop_counter.argtypes = [ctypes.POINTER(PMLibCounter)]

        # # int pm_get_counter_data( counter_t *pm_counter );
        # self._pmlib.pm_get_counter_data.restype = int
        # self._pmlib.pm_get_counter_data.argtypes = [ctypes.POINTER(PMLibCounter)]

        # # int pm_print_data_text(char *file_name,  counter_t pm_counter, line_t lines, int set);
        # self._pmlib.pm_print_data_text.restype = int
        # self._pmlib.pm_print_data_text.argtypes = [ctypes.c_char_p, PMLibCounter, PMLibLines, ctypes.c_int]

        # # int pm_get_counter_data( counter_t *pm_counter );
        # self._pmlib.pm_finalize_counter.restype = int
        # self._pmlib.pm_finalize_counter.argtypes = [ctypes.POINTER(PMLibCounter)]

    def pm_create_server(self, server_ip, port):
        """
        -------
        """
        server = _PMLibServer()
        status = self._pmlib.pm_set_server(server_ip.encode('utf-8'), port, ctypes.byref(server))
        if status == 0:
            return server
        else:
            raise PMLibException("pm_create_server failed!")

    def pm_create_lines(self, lines_string):
        """
        -------
        """
        lines = _PMLibLines()
        status = self._pmlib.pm_set_lines(lines_string.encode('utf-8'), ctypes.byref(lines))
        if status == 0:
            return lines
        else:
            raise PMLibException("pm_create_lines failed!")

    def pm_create_counter(self, counter_string, lines, aggregate, interval, server):
        """
        -------
        """
        counter = _PMLibCounter()
        status = self._pmlib.pm_create_counter(counter_string.encode('utf-8'), lines, aggregate, interval, server,
                                               ctypes.byref(counter))
        if status == 0:
            return counter
        else:
            raise PMLibException("pm_create_counter failed!")

    def pm_start_counter(self, counter):
        """
        ----------
        """
        status = self._pmlib.pm_start_counter(ctypes.byref(counter))
        return status

    def pm_stop_counter(self, counter):
        """
        ----------
        """
        status = self._pmlib.pm_stop_counter(ctypes.byref(counter))
        return status

    def pm_get_counter_data(self, counter):
        """
        ----------
        """
        status = self._pmlib.pm_get_counter_data(ctypes.byref(counter))
        return status

    def pm_print_data_text(self, output_string, counter, lines, set_value):
        """
        -------
        """
        status = self._pmlib.pm_print_data_text(output_string.encode('utf-8'), counter, lines, set_value)
        return status

    def pm_finalize_counter(self, counter):
        """
        ----------
        """
        status = self._pmlib.pm_finalize_counter(ctypes.byref(counter))
        if status == 0:
            return status
        else:
            raise PMLibException("pm_finalize_counter failed!")
