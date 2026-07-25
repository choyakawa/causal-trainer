"""Shared retry helpers for transient Hugging Face network failures."""

from __future__ import annotations

import errno
import logging
import math
import time
from collections.abc import Callable, Iterator
from typing import TypeVar


_Result = TypeVar("_Result")
_LOGGER = logging.getLogger(__name__)
_RETRYABLE_HTTP_STATUS_CODES = {408, 425, 429}
_RETRYABLE_ERRNOS = {
    errno.ECONNABORTED,
    errno.ECONNREFUSED,
    errno.ECONNRESET,
    errno.EHOSTUNREACH,
    errno.ENETDOWN,
    errno.ENETRESET,
    errno.ENETUNREACH,
    errno.EPIPE,
    errno.ETIMEDOUT,
}
_RETRYABLE_EXCEPTION_NAMES = {
    "ChunkedEncodingError",
    "ClientConnectionError",
    "ClientConnectorError",
    "ClientOSError",
    "ClientPayloadError",
    "ContentDecodingError",
    "ConnectError",
    "ConnectTimeout",
    "ConnectionClosed",
    "ConnectionError",
    "ConnectionResetError",
    "DecodingError",
    "FSTimeoutError",
    "IncompleteRead",
    "MaxRetryError",
    "NetworkError",
    "NewConnectionError",
    "PoolTimeout",
    "ProtocolError",
    "ProxyError",
    "ReadError",
    "ReadTimeout",
    "ReadTimeoutError",
    "RemoteDisconnected",
    "RemoteProtocolError",
    "ServerConnectionError",
    "ServerDisconnectedError",
    "ServerTimeoutError",
    "SSLError",
    "Timeout",
    "TimeoutError",
    "TransportError",
    "URLError",
    "WriteError",
    "WriteTimeout",
    "gaierror",
}
_NETWORK_MODULE_ROOTS = (
    "aiohttp",
    "fsspec",
    "httpcore",
    "http",
    "httpx",
    "requests",
    "ssl",
    "urllib",
    "urllib3",
)


def _exception_chain(error: BaseException) -> Iterator[BaseException]:
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _http_status_code(error: BaseException) -> int | None:
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)
    if status is None:
        status = getattr(response, "status", None)
    if status is None:
        status = getattr(error, "status_code", None)
    if status is None:
        status = getattr(error, "status", None)
    if status is None:
        status = getattr(error, "code", None)
    if type(status) is int:
        return status
    return None


def is_retryable_hf_error(error: BaseException) -> bool:
    """Return whether ``error`` represents a transient Hugging Face network failure.

    Authentication, authorization, missing repositories/files, malformed data,
    and application errors deliberately fail fast. Wrapped exceptions are
    inspected through their explicit cause/context chain.
    """

    for candidate in _exception_chain(error):
        status = _http_status_code(candidate)
        if status is not None:
            return status in _RETRYABLE_HTTP_STATUS_CODES or status >= 500

        if isinstance(candidate, (ConnectionError, TimeoutError)):
            return True
        if isinstance(candidate, OSError) and getattr(candidate, "errno", None) in _RETRYABLE_ERRNOS:
            return True

        candidate_type = type(candidate)
        if candidate_type.__name__ == "gaierror":
            return True
        if any(
            any(
                base.__module__ == root
                or base.__module__.startswith(f"{root}.")
                for root in _NETWORK_MODULE_ROOTS
            )
            and base.__name__ in _RETRYABLE_EXCEPTION_NAMES
            for base in candidate_type.__mro__
        ):
            return True
    return False


def _validate_delays(initial_delay: float, max_delay: float) -> tuple[float, float]:
    initial = float(initial_delay)
    maximum = float(max_delay)
    if not math.isfinite(initial) or not math.isfinite(maximum):
        raise ValueError("retry delays must be finite")
    if initial < 0:
        raise ValueError("initial_delay cannot be negative")
    if maximum < initial:
        raise ValueError("max_delay must be greater than or equal to initial_delay")
    return initial, maximum


def retry_hf_call(
    call: Callable[[], _Result],
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    operation: str = "Hugging Face operation",
) -> _Result:
    """Call ``call`` until it succeeds or raises a non-network exception.

    Transient failures use unbounded exponential backoff capped by
    ``max_delay``. ``KeyboardInterrupt`` and other ``BaseException`` subclasses
    are never intercepted.
    """

    if not callable(call):
        raise TypeError("call must be callable")
    initial, maximum = _validate_delays(initial_delay, max_delay)
    delay = initial
    attempt = 0
    while True:
        try:
            return call()
        except Exception as exc:
            if not is_retryable_hf_error(exc):
                raise
            attempt += 1
            _LOGGER.warning(
                "%s failed because of a transient network error; retrying indefinitely "
                "in %.1fs (attempt %d): %s",
                operation,
                delay,
                attempt,
                exc,
            )
            time.sleep(delay)
            delay = min(maximum, max(initial, delay * 2.0))


__all__ = ["is_retryable_hf_error", "retry_hf_call"]
