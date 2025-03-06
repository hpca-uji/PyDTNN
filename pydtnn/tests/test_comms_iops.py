"""Communications IOPS test"""

import os
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
parser = ArgumentParser(prog="test_comms_iops", description="Communications IOPS test")
parser.add_argument("mode", choices=list(Mode))
parser.add_argument("--addr", type=str, default="127.0.0.1")
parser.add_argument("--port", type=int, default=50000)
parser.add_argument("--start-delay", type=float, default=3.0)
parser.add_argument("--end-delay", type=float, default=1.5)
parser.add_argument("--delay", type=float, default=3.0)
parser.add_argument("--size", type=int, default=1_000)
parser.add_argument("--reps", type=int, default=1_000_000)


def server(config: Namespace):
    """Server mode"""
    with comms.Server(addr=config.addr, port=config.port) as server:
        server.get()

        time.sleep(config.delay)
        for i in range(config.reps):
            print(i, end="\r", flush=True)
            msg = server.get().obj
        print()
        for i in range(config.reps):
            print(i, end="\r", flush=True)
            server.put(msg)
        print()

        time.sleep(config.end_delay)


def client(config: Namespace):
    """Client mode"""
    time.sleep(config.start_delay)
    put_msg = os.urandom(config.size)

    with comms.Client(addr=config.addr, port=config.port) as client:
        client.put(None)

        for i in range(config.reps):
            print(i, end="\r", flush=True)
            client.put(put_msg)
        print()
        time.sleep(config.delay)
        for i in range(config.reps):
            print(i, end="\r", flush=True)
            get_msg = client.get().obj
            assert len(put_msg) == len(get_msg), "Lost message data"
        assert put_msg == get_msg, "Corrupted message data"
        print()


def main(config: Namespace):
    """Application entrypoint"""
    self = sys.modules[__name__]
    handler = getattr(self, config.mode)
    print(config)
    handler(config)


if __name__ == "__main__":
    main(parser.parse_args())
