"""MQTT client"""

from concurrent.futures import Future, ThreadPoolExecutor
import uuid
import threading
from queue import SimpleQueue

import paho.mqtt.client as mqtt_client

from pydtnn.comms.mqtt import Protocol
from pydtnn.utils.io_stream import Stream
from pydtnn.utils import UUID_MAX, UUID_NIL
from pydtnn.comms import ConnectionData, ConnectionState, ResourceClosed, Message


__all__ = (
    "Client",
)


# Sentinel objects
END_COMM = b""


class Client(Protocol):
    """MQTT client"""

    def __init__(self, addr: str, port: int) -> None:
        """Client initialization"""
        super().__init__(addr, port)

        # State
        self._lock = threading.Condition()
        self._get_event = SimpleQueue[uuid.UUID]()
        self._state = ConnectionData(buffer_size=self._max_message_size)
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"{__name__}.{self.__class__.__qualname__}:{id(self)}")

        # MQTT
        self._register_handler(topic=f"s2c/{self._id.hex}", handler=self._handle_message)

        self._session_ini()

    def _session_ini(self) -> None:
        """Send session ini message"""
        state = self._state
        assert ConnectionState.WRITABLE not in state.state, "Sending session ini on writable stream"
        state.state |= ConnectionState.WRITABLE
        self.put(self._id)

    def _session_fin(self) -> None:
        """Send session fin message"""
        state = self._state
        assert ConnectionState.WRITABLE in state.state, "Sending session fin on unwritable stream"
        state.state &= ~ConnectionState.WRITABLE
        self._put(Stream())

    def _handle_session_ini(self, stream: Stream) -> None:
        """Handle session initialize message"""
        state = self._state
        assert ConnectionState.READABLE not in state.state, "Recived session ini on readable stream"

        # Set peer in state
        with stream:
            id = self._serializer.load(stream)
        state.peer = id
        state.state |= ConnectionState.READABLE

    def _handle_session_fin(self, stream: Stream) -> None:
        """Handle session finalize message"""
        state = self._state
        assert ConnectionState.READABLE in state.state, "Recived session fin on unreadable stream"
        stream.close()
        state.state &= ~ConnectionState.READABLE

    def _handle_message(self, client: mqtt_client.Client, userdata, message: mqtt_client.MQTTMessage) -> None:
        """Broker message handler"""
        state = self._state
        self._state.get_buffer.write(message.payload)
        self._get_flush()

        if not state.state and state.put_empty():
            self._fin()

    def _fin(self) -> None:
        """Communication finalization"""
        with self._lock:
            # del self._client
            self._lock.notify_all()

    def _get_flush(self) -> None:
        state = self._state
        peer = state.peer

        while True:
            try:
                stream = state.get()
            except BlockingIOError:
                break

            if stream.empty():
                self._handle_session_fin(stream)

            elif state.peer == UUID_NIL:
                self._handle_session_ini(stream)
                peer = state.peer

            else:
                state.get_queue.put(stream)
                self._get_event.put(peer)

    def _put(self, stream: Stream) -> Future[None]:
        """Put stream into queue and notify"""
        future = self._state.put(stream)
        self._submit(self._c2s)
        return future

    def _c2s(self):
        state = self._state

        state.put_flush()
        if state.put_buffer.empty():
            return
        with state.put_read() as view:
            self._publish(f"c2s/{self._id.hex}", bytes(view))

    def put(self, obj, *peers: uuid.UUID) -> Future[None]:
        """Publish data to server"""
        assert len(peers) == 0, "Client can not publish to another client"
        stream = self._serializer.dump(obj)
        return self._put(stream)

    def get(self, *peers: uuid.UUID) -> Message:
        """Get from the server"""
        assert len(peers) == 0, "Client can not get from another client"
        peer = self._get_event.get()

        # Exit signaled
        if peer == UUID_MAX:
            raise ResourceClosed()

        state = self._state
        get_queue = state.get_queue

        # Get object
        stream = get_queue.get_nowait()

        with stream:
            obj = self._serializer.load(stream)

        return Message(peer=state.peer, obj=obj)

    def _close(self) -> None:
        """Close the client"""
        self._session_fin()

        # Request loop thread to stop
        self._pool.shutdown()
        self._stop_loop()

        # Unlock inflight external API:
        for _ in range(threading.active_count()):
            self._get_event.put(UUID_MAX)

        # Close resources
        super()._close()
