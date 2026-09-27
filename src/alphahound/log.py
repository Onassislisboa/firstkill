"""Structured logging.

JSON lines to stdout so the output is greppable after the fact, which matters
more than it sounds: when a trade goes wrong you are reconstructing a sequence
of events from four subsystems, and prose logs make that archaeology.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import queue
import sys
from typing import Any

_RESERVED = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "message",
    "asctime",
    "taskName",
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "t": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class _DropQueueHandler(logging.handlers.QueueHandler):
    """Never block the caller. A full queue means the listener is stuck on the pipe."""

    def enqueue(self, record: logging.LogRecord) -> None:
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            pass


def setup(level: str = "INFO") -> None:
    # ponytail: one listener thread. A dead terminal fills the stdout pipe and
    # StreamHandler.write blocks; that used to freeze the whole asyncio loop
    # for hours with no exception. The listener may block; the bot must not.
    # Ceiling: 1000 queued lines, then drops. A file handler if you need the rest.
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(JsonFormatter())
    handler = _DropQueueHandler(queue.Queue(maxsize=1000))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.handlers.QueueListener(handler.queue, stream, respect_handler_level=True).start()


def get(name: str) -> logging.Logger:
    return logging.getLogger(f"alphahound.{name}")
