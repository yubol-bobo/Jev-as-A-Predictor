"""Localhost-only network guard (PhysioNet and Redivis DUA compliance). Copied from third_party/clinenv/env/local_guard.py.

MIMIC-derived content must never leave the cluster. This module is imported
before anything else in every ClinEnv entry point (via env/__init__.py and the
top of each CLI script). On import it:

  1. removes OPENAI_API_KEY / ANTHROPIC_API_KEY / DEEPSEEK_API_KEY from the
     process environment (so no SDK can silently fall back to a cloud API);
  2. forces Hugging Face libraries offline (HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE);
  3. monkey-patches socket.socket.connect / connect_ex so that any outbound
     TCP/UDP connection to a non-loopback address raises LocalOnlyViolation.
     This catches every library (openai, anthropic, httpx, requests, urllib,
     pyhealth downloads, ...) inside this Python process;
  4. provides assert_local_url() so clients fail early, with a clear message,
     when configured with a non-localhost base_url.

Escape hatch: S1CLIN_ALLOW_NETWORK=1 disables (3) only. Use it only for jobs
that touch NO MIMIC data (e.g. a one-off fetch of public ICD/ATC vocabularies).
"""

from __future__ import annotations

import ipaddress
import os
import socket
import sys
from urllib.parse import urlparse

_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}
_BLOCKED_KEYS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY",
                 "GEMINI_API_KEY", "GOOGLE_API_KEY", "HF_TOKEN")


class LocalOnlyViolation(RuntimeError):
    """Raised when code tries to reach a non-loopback network address."""


def _is_loopback(host) -> bool:
    if host is None:
        return False
    if isinstance(host, bytes):
        host = host.decode(errors="ignore")
    host = str(host).strip("[]")
    if host in _LOCAL_HOSTS:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def assert_local_url(url: str | None, what: str = "endpoint") -> None:
    """Raise LocalOnlyViolation unless url points at a loopback host."""
    if not url:
        raise LocalOnlyViolation(
            f"{what}: no base_url configured; refusing to fall back to a cloud API default.")
    host = urlparse(url).hostname
    if not _is_loopback(host):
        raise LocalOnlyViolation(
            f"{what}: base_url {url!r} is not localhost. MIMIC-derived content may only "
            f"be sent to 127.0.0.1 endpoints (PhysioNet DUA).")


def _scrub_env() -> list[str]:
    removed = []
    for k in _BLOCKED_KEYS:
        if os.environ.pop(k, None) is not None:
            removed.append(k)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    return removed


_orig_connect = socket.socket.connect
_orig_connect_ex = socket.socket.connect_ex


def _check_addr(sock, address):
    fam = getattr(sock, "family", None)
    if fam not in (socket.AF_INET, socket.AF_INET6):
        return  # AF_UNIX etc. are local by construction
    host = address[0] if isinstance(address, tuple) else address
    if not _is_loopback(host):
        raise LocalOnlyViolation(
            f"Blocked outbound connection to {address!r}: only loopback is allowed "
            f"while processing MIMIC data (set S1CLIN_ALLOW_NETWORK=1 only for jobs "
            f"that touch no MIMIC data).")


# Optional audit log of every connection attempt (S1CLIN_LOG_CONNECTIONS=<path>):
# one line "family host port" per connect, used by the planner dry-run to prove that
# all traffic went to 127.0.0.1:8011.
_CONN_LOG = os.environ.get("S1CLIN_LOG_CONNECTIONS")


def _log_conn(sock, address):
    if not _CONN_LOG:
        return
    try:
        fam = getattr(sock, "family", None)
        name = {socket.AF_INET: "inet", socket.AF_INET6: "inet6"}.get(fam, "other")
        with open(_CONN_LOG, "a") as f:
            f.write(f"{name} {address!r}\n")
    except OSError:
        pass


def _guarded_connect(self, address):
    _log_conn(self, address)
    _check_addr(self, address)
    return _orig_connect(self, address)


def _guarded_connect_ex(self, address):
    _log_conn(self, address)
    _check_addr(self, address)
    return _orig_connect_ex(self, address)


def install() -> None:
    removed = _scrub_env()
    if removed:
        print(f"[local_guard] removed {', '.join(removed)} from environment", file=sys.stderr)
    if os.environ.get("S1CLIN_ALLOW_NETWORK") == "1":
        print("[local_guard] WARNING: S1CLIN_ALLOW_NETWORK=1, socket guard disabled", file=sys.stderr)
        return
    if socket.socket.connect is not _guarded_connect:
        socket.socket.connect = _guarded_connect
        socket.socket.connect_ex = _guarded_connect_ex


def is_installed() -> bool:
    return socket.socket.connect is _guarded_connect


install()
