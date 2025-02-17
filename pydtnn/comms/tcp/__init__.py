"""TCP communications"""

# NOTE: Module considerations
#
# Experimental and unbuffered


import struct
import socket
import functools

from pydtnn import comms


__all__ = (
    "Connection",
    "Protocol"
)


class Connection:
    _format_size = "!i"
    _sizeof_size = struct.calcsize(_format_size)
    _recv_flags = socket.MSG_WAITALL if hasattr(socket, "MSG_WAITALL") else 0

    def __init__(self, socket: socket.socket) -> None:
        self.closed = False
        self._socket = socket

    @functools.cached_property
    def _peer(self):
        return self._socket.getpeername()

    @property
    def _addr(self) -> str:
        return self._peer[0]

    @property
    def _port(self) -> int:
        return self._peer[1]

    def __enter__(self):
        """Context manager start"""
        return self

    def __exit__(self, cls, exc, tb):
        """Context manager exit"""
        self.close()

    @property
    def _netloc(self):
        return f"{self._addr}:{self._port}"

    def fileno(self) -> int:
        return self._socket.fileno()

    def send(self, data: bytes) -> None:
        try:
            self._socket.sendall(data)
        except OSError:
            raise comms.ResourceClosed()

    def recv(self, size: int) -> bytes:
        buffer = bytearray(size)

        while size > 0:
            recv = self._socket.recv_into(buffer, flags=self._recv_flags)
            if recv:
                size -= recv
            else:
                raise comms.ResourceClosed()

        return bytes(buffer)

    def get(self) -> bytes:
        data = self.recv(self._sizeof_size)
        size, = struct.unpack(self._format_size, data)
        return self.recv(size)

    def put(self, data: bytes) -> None:
        size = struct.pack(self._format_size, len(data))
        self.send(size)
        self.send(data)

    def close(self):
        if self.closed:
            return
        self.closed = True
        self._socket.shutdown(socket.SHUT_WR)
        self._socket.close()

    def __del__(self) -> None:
        """Best effort finalizer"""
        try:
            self.close()
        except:  # noqa: E722
            pass


class Protocol(comms.Communication):
    """Shared base TCP implementation"""
    _poll_interval = 0.5
