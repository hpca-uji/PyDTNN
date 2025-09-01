"""MQTT client"""

import uuid
import threading
from concurrent.futures import Future

import paho.mqtt.client as mqtt_client

from pydtnn.comms import client
from pydtnn.utils import UUID_MAX
from pydtnn.comms.mqtt import Protocol
from pydtnn.utils.io_stream import Stream
from pydtnn.comms import CommunicatorOptions, ResourceClosed, Message


__all__ = (
    "Client",
)


# Sentinel objects
END_COMM = b""


class Client(Protocol, client.Client):
    """MQTT client"""

    def __init__(self, options: CommunicatorOptions = CommunicatorOptions()) -> None:
        """Client initialization"""
        super().__init__(options)

        # MQTT
        self._register_handler(topic=f"s2c/{self._id.hex}", handler=self._handle_message)

        self._put(self._session_ini())

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

    def _put(self, stream: Stream) -> Future[None]:
        """Put stream into queue and notify"""
        future = self._state.put(stream)
        self._pool.submit(self._c2s).add_done_callback(lambda future: future.result())
        return future

    def _c2s(self):
        state = self._state

        state.put_flush()
        while not state.put_buffer.empty():
            with state.put_read(self._options.connection.max_size) as view:
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
        self._put(self._session_fin())

        # Request loop thread to stop
        self._pool.shutdown()
        self._stop_loop()

        # Unlock inflight external API:
        for _ in range(threading.active_count()):
            self._get_event.put(UUID_MAX)

        # Close resources
        super()._close()
