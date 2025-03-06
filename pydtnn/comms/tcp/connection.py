"""Message-oriended TCP connection"""

# NOTE: Message format used for commuications:
# - Network byte order (big-endain)
#
# +---------------+-----------------+
# | Size (uint32) | Data (variable) |
# +---------------+-----------------+


import socket
import struct
import functools
from queue import Empty

from pydtnn import comms


class Connection:
    """
    Message-oriended TCP connection

    Use selectors to track file descriptor availability,
    then call recv and send when apropiate to process the buffers.

    Blocking interface: put, get
    Non-blocking interface: put_nowait, get_nowait (raise queue.Empty)
    """

    _format_size = "!I"
    _sizeof_size = struct.calcsize(_format_size)

    def __init__(self, sock: socket.socket) -> None:
        """Inizialie connection metadata"""
        self.closed = False
        self._socket = sock

        self._send_queue = bytearray()
        self._recv_queue = bytearray()

        size = self._socket.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
        self._buffer = bytearray(size)

    def recv(self) -> None:
        """Process a recive request"""
        # Recive data to pre-allocated buffer
        size = self._socket.recv_into(self._buffer)

        # Append recived slice to queue without a intermediate copy
        if size > 0:
            with memoryview(self._buffer) as view:
                self._recv_queue.extend(view[:size])

        # Empty recive signals EOF, but wait for queue
        if len(self._recv_queue) == 0 and size == 0:
            raise comms.ResourceClosed()

    def send(self) -> None:
        """Process a send request"""
        # Prevent empty send
        if len(self._send_queue) == 0:
            return

        # Send data and reduce queue in-place
        size = self._socket.send(self._send_queue)
        del self._send_queue[:size]

        # Empty send signals EOF, queue is lost
        if size == 0:
            del self._send_queue[:]
            raise comms.ResourceClosed()

    def get_nowait(self) -> bytes:
        """Attempt to get a message without waiting (raises Empty)"""
        # Check if message size available
        size = self._sizeof_size
        if len(self._recv_queue) < size:
            raise Empty()

        # Check if message data available
        size += struct.unpack(self._format_size, self._recv_queue[:self._sizeof_size])[0]
        if len(self._recv_queue) < size:
            raise Empty()

        # Cast message slice to bytes without a intermediate copy, and reduce queue in-place
        with memoryview(self._recv_queue) as view:
            data = view[self._sizeof_size:size].tobytes()
        del self._recv_queue[:size]

        return data

    def get(self) -> bytes:
        """Get a message (possibly blocking)"""
        # Process recive request until a message is available
        while True:
            try:
                data = self.get_nowait()
            except Empty:
                self.recv()
                continue
            else:
                break
        return data

    def put_nowait(self, data: bytes) -> None:
        """Publish a message without waiting to underling send"""
        size = struct.pack(self._format_size, len(data))
        self._send_queue.extend(size)
        self._send_queue.extend(data)

    def put(self, data: bytes) -> None:
        """Publish a message and ensure it is send (possibly blocking)"""
        self.put_nowait(data)
        self._flush()

    def _flush(self) -> None:
        """Flush send queue"""
        while self._send_queue:
            self.send()

    def close(self):
        """Close the connection"""
        if self.closed:
            return
        self.closed = True
        self._flush()
        self._socket.close()

    @functools.cached_property
    def peer(self) -> str:
        """Peer identification"""
        return "{}:{}".format(*self._socket.getpeername())

    def fileno(self) -> int:
        """File descriptor"""
        return self._socket.fileno()

    def __del__(self) -> None:
        """Best effort finalizer"""
        try:
            self.close()
        except:  # noqa: E722
            pass
