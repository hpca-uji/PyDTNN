"""Runtime configuration options"""

import os
from pathlib import Path

from net_queue import Protocol, CommunicatorOptions


__all__ = (
    "init",
    "wait",
    "addr",
    "port",
    "size",
    "rank",
    "serial",
    "proto",
    "ssl",
    "ssl_key",
    "ssl_cert",
    "comm"
)


"""Should server auto initialize"""
init = bool(
    not os.environ.get("PYMPI_ADDR")
)

"""Wait for auto server transitions"""
wait = float(
    os.environ.get("PYMPI_WAIT")
    or 0.5
)


"""Server address"""
addr = (
    os.environ.get("PYMPI_ADDR")
    or "127.0.0.1"
)


"""Server port"""
port = int(
    os.environ.get("PYMPI_PORT")
    or 61642
)


"""Communication size"""
size = int(
    os.environ.get("PYMPI_SIZE")
    or os.environ.get("OMPI_COMM_WORLD_SIZE")
    or os.environ.get("PMI_SIZE")
    or os.environ.get("SLUM_NPROCS")
    or 1
)


"""Communication identifier"""
rank = int(
    os.environ.get("PYMPI_RANK")
    or os.environ.get("OMPI_COMM_WORLD_RANK")
    or os.environ.get("PMI_RANK")
    or os.environ.get("SLUM_PROCID")
    or 0
)

"""Serializable global names"""
serial = list(filter(None, (
    os.environ.get("PYMPI_SERIAL")
    or ""
).split(",")))

"""Communication protocol"""
proto = Protocol(
    os.environ.get("PYMPI_PROTO")
    or "tcp"
)

"""Use secure communications"""
ssl = bool(
    os.environ.get("PYMPI_SSL")
    or False
)


"""Secure communications private key"""
ssl_key = (
    Path(key)
    if (key := os.environ.get("PYMPI_SSL_KEY"))
    else None
)

"""Secure communications certificate chain"""
ssl_cert = (
    Path(key)
    if (key := os.environ.get("PYMPI_SSL_CERT"))
    else None
)

"""Additional net-queue communicator options"""
comm = CommunicatorOptions()
