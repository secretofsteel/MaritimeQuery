"""Real-time log streaming via SSE."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import Set, List

# Ring buffer — keeps last N log lines in memory
LOG_BUFFER: deque = deque(maxlen=500)

# Connected SSE clients — each is an asyncio.Queue
_CLIENTS: Set[asyncio.Queue] = set()


class BroadcastLogHandler(logging.Handler):
    """Logging handler that pushes formatted records to all SSE clients
    and stores them in a ring buffer for new clients to catch up.

    Thread-safe: logging can happen from background processing threads.
    The handler uses asyncio call_soon_threadsafe to push to async queues.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            entry = self.format(record)
            LOG_BUFFER.append(entry)

            # Push to all connected async clients (thread-safe)
            if not _CLIENTS:
                return

            loop = asyncio.get_event_loop()
            if loop.is_running():
                for queue in list(_CLIENTS):
                    try:
                        # Use call_soon_threadsafe because log calls come from
                        # background threads (processing, indexing, etc.)
                        loop.call_soon_threadsafe(queue.put_nowait, entry)
                    except Exception:
                        pass  # Client disconnected or queue full — skip
        except Exception:
            self.handleError(record)


def get_log_buffer() -> List[str]:
    """Return current buffer contents (for new client catchup)."""
    return list(LOG_BUFFER)


def register_client() -> asyncio.Queue:
    """Register a new SSE client. Returns a queue to read from."""
    queue = asyncio.Queue(maxsize=100)
    _CLIENTS.add(queue)
    return queue


def unregister_client(queue: asyncio.Queue) -> None:
    """Remove a disconnected client."""
    _CLIENTS.discard(queue)


def install_log_handler() -> None:
    """Attach the broadcast handler to app and API loggers.

    Attaches to specific loggers rather than just root, because the
    main app logger (maritime_rag) has propagate=False and won't
    forward records to root.
    """
    handler = BroadcastLogHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    ))

    # Target loggers — attach to each one that matters
    target_loggers = [
        "",                 # root — catches uvicorn, third-party libs
        "maritime_rag",     # main app logger (propagate=False)
        "api",              # API route loggers (api.routes.documents etc.)
    ]

    for logger_name in target_loggers:
        lg = logging.getLogger(logger_name)
        # Avoid duplicate handlers on reload
        if not any(isinstance(h, BroadcastLogHandler) for h in lg.handlers):
            lg.addHandler(handler)
