"""TCP communications"""

# NOTE: Module considerations
#
# Experimental

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
    _buffer_size = 2 ** 20  # 1 MB

    def __init__(self, sock: socket.socket) -> None:
        self.closed = False
        self._socket = sock
        self._send_buffer = bytearray()
        self._recv_buffer = bytearray()
        self._temp_buffer = bytearray(self._buffer_size)

    def recv(self) -> None:
        size = self._socket.recv_into(self._temp_buffer)
        if size > 0:
            with memoryview(self._temp_buffer) as view:
                self._recv_buffer.extend(view[:size])

        if len(self._recv_buffer) == 0 and size == 0:
            raise comms.ResourceClosed()

    def send(self) -> None:
        if len(self._send_buffer) == 0:
            return

        size = self._socket.send(self._send_buffer)
        del self._send_buffer[:size]

        if len(self._send_buffer) == 0 and size == 0:
            raise comms.ResourceClosed()

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
        with memoryview(self._recv_buffer) as view:
            data = view[self._sizeof_size:size].tobytes()
        del self._recv_buffer[:size]

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

    @functools.cached_property
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
