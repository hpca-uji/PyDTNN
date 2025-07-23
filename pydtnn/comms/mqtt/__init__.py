"""MQTT communications"""

# NOTE: MQTT broker implementations are not common, so the server provided here
# is actually another client. Therefore the address and port provided to both,
# the client and server, should be the one of the actual broker, not where the
# server is running.

# NOTE: The MQTT library handles comunications single-threaded, therefore
# operations on related callbacks are limited to pushing or pulling data from
# queues without blocking, so all operations are minimal and fast.

# FIXME: Peer-groups and global comunications are not optimized. First, chunked
# message ordering must be resolved. Single chunk order it is guaranteed by
# the protocol, even on with diferent topics. Second, peer-groups could be
# implemented using grouping requests that generate new UUID per group.
# This would reduce also reduce load on the broker.

from concurrent.futures import ThreadPoolExecutor

import paho.mqtt.enums as mqtte_enum
import paho.mqtt.client as mqtt_client

from pydtnn import comms
from pydtnn.utils import UUID_NIL


__all__ = (
    "Protocol",
)


# Sentinel object
ARG_MISSING = object()


class Protocol(comms.Communicator):
    """Shared base MQTT implementation"""

    _qos = 0
    _transport = "tcp"
    _protocol = mqtt_client.MQTTv311

    def __init__(self, options: comms.CommunicatorOptions = {}) -> None:
        """Communication initialization"""
        super().__init__({**options, "workers": 1})

        # State
        self._ack_queue = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"{__name__}.{self.__class__.__qualname__}:{id(self)}.acks")

        # MQTT
        self._client = mqtt_client.Client(
            callback_api_version=mqtte_enum.CallbackAPIVersion.VERSION2,
            client_id=self._id.hex,
            protocol=self._protocol,
            transport=self._transport  # type: ignore
        )

        if comms.SSL:
            self._client.tls_set(ca_certs=str(comms.SSL_CERT) if comms.SSL_CERT else None)

        self._client.connect(host=self._addr, port=self._port)
        self._client.loop_start()

    def _submit(self, fn, /, *args, **kwargs):
        """Process in the pool with exception handeling"""
        future = self._pool.submit(fn, *args, **kwargs)
        future.add_done_callback(lambda future: future.result())
        return future

    def _start_loop(self) -> None:
        """Start connection handling loop"""
        self._client.loop_start()

    def _stop_loop(self) -> None:
        self._ack_queue.shutdown()
        self._client.loop_stop()

    def _register_handler(self, topic: str, handler: mqtt_client.CallbackOnMessage) -> None:
        """Setup a topic handler"""
        self._client.message_callback_add(sub=topic, callback=handler)
        self._client.subscribe(topic=topic, qos=self._qos)

    def _peer(self, message: mqtt_client.MQTTMessage) -> str:
        """Get peer from a mesage"""
        return message.topic.split("/", 1)[1]

    def _publish(self, topic: str, data=None) -> None:
        """Generic MQTT publish"""
        message = self._client.publish(topic=topic, payload=data, qos=self._qos)
        self._ack_queue.submit(message.wait_for_publish).add_done_callback(lambda future: future.result())
