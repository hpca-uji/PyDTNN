"""MQTT communications"""


from pydtnn import comms
import paho.mqtt.client as mqtt_client


__all__ = (
    "Protocol",
)


class Protocol(comms.Communication):
    """Shared base MQTT implementation"""

    _qos = 0
    _transport = "tcp"
    _protocol_port = 1883
    _protocol = mqtt_client.MQTTv311
