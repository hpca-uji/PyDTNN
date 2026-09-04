"""Logging related utilities"""

import logging
from pathlib import Path
from typing import Any

from pydtnn import package_name, timestamp
from pydtnn.utils import logger


class TqdmLogger:
    """Logger wrapper to redirect tqdm output to logging system."""

    def __init__(self, csi: bool = True) -> None:
        """Pad string to the logger."""
        self._csi = csi
        if self._csi:
            logger.info(self.__class__.__name__)

    def write(self, s: str) -> int:
        """Write string to the logger."""
        if not self._csi:
            s = s.strip("\r")
        if s := s.replace("\r", "\x1b[F").replace("\n", ""):
            logger.info(s.replace("\r", "\x1b[F").replace("\n", ""))
        return len(s)


class TimestampedFileHandler(logging.FileHandler):
    """Timestap-based logging file handler."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize file logger"""
        path = Path(f"{package_name}-{timestamp}.log").resolve()
        super().__init__(filename=path, mode="w", *args, **kwargs)
