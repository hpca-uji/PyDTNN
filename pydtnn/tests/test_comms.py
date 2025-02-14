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
parser.add_argument("--addr", type=str, default="127.0.0.1")
parser.add_argument("--port", type=int, default=50000)
parser.add_argument("--size", type=int, default=1)
parser.add_argument("--delay", type=float, default=3.0)
parser.add_argument("--hold", type=float, default=1.5)


def server(config: Namespace):
    """Server mode"""
    server = comms.Server(addr=config.addr, port=config.port)
    messages = []

    for _ in range(config.size):
        client_msg = server.get()
        print(f"server[{server.id}]-c2s: {client_msg}")
        messages.append(client_msg)

    server_msg = server.id
    print(f"server[{server.id}]-s2c-global: {server_msg}")
    server.put(obj=server_msg)

    for server_msg in messages:
        print(f"server[{server.id}]-s2c-local: {server_msg}")
        server.put(server_msg.obj, server_msg.peer)

    time.sleep(config.hold)
    server.close()


def client(config: Namespace):
    """Client mode"""
    time.sleep(config.delay)
    client = comms.Client(addr=config.addr, port=config.port)

    client_msg = client.id
    print(f"client[{client.id}]-c2s: {client_msg}")
    client.put(client_msg)

    server_msg = client.get()
    print(f"client[{client.id}]-s2c-global: {server_msg}")

    server_msg = client.get()
    print(f"client[{client.id}]-s2c-local: {server_msg}")
    client.close()


def main(config: Namespace):
    """Application entrypoint"""
    self = sys.modules[__name__]
    handler = getattr(self, config.mode)
    handler(config)


if __name__ == "__main__":
    main(parser.parse_args())
