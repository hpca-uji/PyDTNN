"""Communications test"""

import sys
import time
import enum
from pydtnn import comms
from argparse import ArgumentParser, Namespace


__all__ = ()


class Mode(enum.StrEnum):
    """Test modes"""
    SERVER = enum.auto()
    CLIENT = enum.auto()


# Argument pasrser
parser = ArgumentParser(prog="test_comms", description="Communications test")
parser.add_argument("mode", choices=list(Mode))
parser.add_argument("--delay", type=float, default=3.0)
parser.add_argument("--hold", type=float, default=1.5)


def server(config: Namespace):
    """Server mode"""
    server = comms.Server()
    client_msg = server.get()
    print(f"server-c2s: {client_msg}")
    server_msg = "global"
    print(f"server-s2c: {server_msg}")
    server.put(obj=server_msg)
    server_msg = comms.Message(peer=client_msg.peer, obj="local")
    print(f"server-s2c: {server_msg}")
    server.put(server_msg.obj, server_msg.peer)
    time.sleep(config.hold)
    server.close()


def client(config: Namespace):
    """Client mode"""
    time.sleep(config.delay)
    client = comms.Client()
    client_msg = "client"
    print(f"client-c2s: {client_msg}")
    client.put(client_msg)
    server_msg = client.get()
    print(f"client-s2c: {server_msg}")
    server_msg = client.get()
    print(f"client-s2c: {server_msg}")
    client.close()


def main(config: Namespace):
    """Application entrypoint"""
    self = sys.modules[__name__]
    handler = getattr(self, config.mode)
    handler(config)


if __name__ == "__main__":
    main(parser.parse_args())
