"""MQTT client"""

import threading
import uuid
from queue import Empty, SimpleQueue

import paho.mqtt.client as mqtt_client

from pydtnn.comms.mqtt import Protocol
from pydtnn.comms import ResourceClosed, Message
from pydtnn.utils.io_stream import StreamSerializer


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
        self._get_event = SimpleQueue()
        self._put_event = threading.Event()

        self._get_queue = SimpleQueue()
        self._put_queue = SimpleQueue()
        self._raw_serialzier = StreamSerializer()

        self._msg_queue = SimpleQueue[mqtt_client.MQTTMessage]()

        # MQTT
        self._start_loop()
        self._register_handler(topic=f"s2c/{self._id.hex}", handler=self._handle_message)
        self._submit(self._ini).result()

    def _handle_message(self, client: mqtt_client.Client, userdata, message: mqtt_client.MQTTMessage) -> None:
        """Broker message handler"""
        self._msg_queue.put(message)

    def _ini(self) -> None:
        self._put_queue.put(self._id)
        self._c2s(method="ini")
        self._server = self._get_queue.get_nowait()

    def _fin(self) -> None:
        self._put_queue.put(self._id)
        self._submit(self._c2s, method="fin").result()
        self._server = self._get_queue.get_nowait()

    def _s2c(self) -> None:
        messages = self._consume_queue(self._msg_queue)

        for message in messages:
            self._raw_serialzier.write(message.payload)
            try:
                while True:
                    self._get_queue.put(self._raw_serialzier.load())
            except BlockingIOError:
                pass

    def _c2s(self, method: str = "c2s"):
        serializer = StreamSerializer()
        buffer = bytearray(self._max_data_size)
        objects = self._consume_queue(self._put_queue)

        # Try to generate full messages
        for obj in objects:
            serializer.dump(obj)
            while serializer.nbytes >= len(buffer):
                size = serializer.readinto(buffer)
                assert size == len(buffer), "Sending partial message"
                self._publish(topic=f"{method}/{self._id}", data=bytes(buffer))

        # Drain serializer
        try:
            size = serializer.readinto(buffer)
        except BlockingIOError:
            return
        with memoryview(buffer) as view:
            with view[:size] as subview:
                self._publish(topic=f"{method}/{self._id}", data=bytes(subview))

    def put(self, obj, *peers: uuid.UUID) -> None:
        """Publish data to server"""
        super().put(obj, *peers)
        assert len(peers) == 0, "Client can not publish to another client"
        self._put_queue.put(obj)
        self._put_event.set()
        self._submit(self._c2s)

    def get(self, *peers: uuid.UUID) -> Message:
        """Get data from the server"""
        super().get(*peers)
        assert len(peers) == 0, "Client can not get from another client"

        try:
            obj = self._get_queue.get_nowait()
        except Empty:
            if hasattr(self, "_client"):
                self._submit(self._s2c)
                obj = self._get_queue.get()
                self._get_event.put(None)
            else:
                raise ResourceClosed()

        return Message(peer=self._server, obj=obj)

    def close(self) -> None:
        """Close the client"""
        if self.closed:
            return
        super().close()

        # Unlock inflight external API

        # Request loop thread to stop
        self._stop_loop()

        # Close resources
        self._pool.shutdown()
