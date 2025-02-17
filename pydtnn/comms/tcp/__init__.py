"""TCP communications"""

# NOTE: Module considerations
#
# Experimental

import io
import struct
import socket
import functools
from queue import Empty

from pydtnn import comms


__all__ = (
    "Connection",
    "Protocol"
)


class Connection:
    _format_size = "!i"
    _sizeof_size = struct.calcsize(_format_size)
    _buffer_size = io.DEFAULT_BUFFER_SIZE

    def __init__(self, socket: socket.socket) -> None:
        self.closed = False
        self._socket = socket
        self._send_buffer = bytearray()
        self._recv_buffer = bytearray()

    def recv(self) -> None:
        data = self._socket.recv(self._buffer_size)
        if data:
            self._recv_buffer.extend(data)

        if len(self._recv_buffer) == 0 and len(data) == 0:
            raise comms.ResourceClosed()

    def send(self) -> None:
        if len(self._send_buffer) == 0:
            return

        size = self._socket.send(self._send_buffer)

        if len(self._recv_buffer) == 0 and size == 0:
            raise comms.ResourceClosed()

        self._send_buffer = self._send_buffer[size:]

    def get_nowait(self) -> bytes:
        # Try size
        size = self._sizeof_size
        if len(self._recv_buffer) < size:
            raise Empty()

        # Try message
        size += struct.unpack(self._format_size, self._recv_buffer[:self._sizeof_size])[0]
        if len(self._recv_buffer) < size:
            raise Empty()

        # Save message
        data = self._recv_buffer[self._sizeof_size:size]
        self._recv_buffer = self._recv_buffer[size:]
        return data

    def get(self) -> bytes:
        while True:
            self.recv()
            try:
                data = self.get_nowait()
            except Empty:
                continue
            else:
                break
        return data

    def put_nowait(self, data: bytes) -> None:
        size = struct.pack(self._format_size, len(data))
        self._send_buffer.extend(size)
        self._send_buffer.extend(data)

    def put(self, data: bytes) -> None:
        self.put_nowait(data)
        while self._send_buffer:
            self.send()

    def close(self):
        if self.closed:
            return
        self.closed = True
        self._socket.shutdown(socket.SHUT_WR)
        self._socket.close()

    @functools.cache
    def peer(self) -> str:
        return "{}:{}".format(*self._socket.getpeername())

    def fileno(self) -> int:
        return self._socket.fileno()

    def __del__(self) -> None:
        """Best effort finalizer"""
        try:
            self.close()
        except:  # noqa: E722
            pass


class Protocol(comms.Communication):
    """Shared base TCP implementation"""
    _poll_interval = 0.5
