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
parser.add_argument("--size", type=int, default=100)
parser.add_argument("--delay", type=float, default=3.0)
parser.add_argument("--hold", type=float, default=1.5)


def server(config: Namespace):
    """Server mode"""
    server = comms.Server(addr=config.addr, port=config.port)

    print("small")
    time.sleep(config.hold)
    for _ in range(config.size):
        print("r")
        msg = server.get().obj
    for _ in range(config.size):
        print("s")
        server.put(msg)
    print()

    print("large")
    msg = server.get().obj
    server.put(msg)

    time.sleep(config.hold)
    server.close()


def client(config: Namespace):
    """Client mode"""
    time.sleep(config.delay)
    client = comms.Client(addr=config.addr, port=config.port)

    print("small")
    msg = client.id
    for _ in range(config.size):
        print("s")
        client.put(msg)
    time.sleep(config.hold)
    for _ in range(config.size):
        print("r")
        msg = client.get().obj
    print()

    print("large")
    msg = [client.id for _ in range(config.size)]
    client.put(msg)
    msg = client.get().obj

    client.close()


def main(config: Namespace):
    """Application entrypoint"""
    self = sys.modules[__name__]
    handler = getattr(self, config.mode)
    handler(config)


if __name__ == "__main__":
    main(parser.parse_args())
