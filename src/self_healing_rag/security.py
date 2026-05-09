from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse


class UnsafeUrlError(ValueError):
    """Raised when a URL points at a blocked target."""


def is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def validate_public_url(url: str, *, allow_private_urls: bool = False) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise UnsafeUrlError("Only http and https URLs are supported.")
    if not parsed.hostname:
        raise UnsafeUrlError("URL must include a hostname.")
    if parsed.username or parsed.password:
        raise UnsafeUrlError("URLs with embedded credentials are not supported.")
    if allow_private_urls:
        return url

    hostname = parsed.hostname.strip().lower()
    if hostname == "localhost" or hostname.endswith(".localhost"):
        raise UnsafeUrlError("Localhost URLs are blocked.")

    try:
        addresses = socket.getaddrinfo(hostname, parsed.port or _default_port(parsed.scheme), type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise UnsafeUrlError(f"Could not resolve URL hostname: {hostname}") from exc

    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if _is_blocked_ip(ip):
            raise UnsafeUrlError(f"Private or local network URL targets are blocked: {ip}")

    return url


def _default_port(scheme: str) -> int:
    return 443 if scheme == "https" else 80


def _is_blocked_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return any(
        (
            ip.is_private,
            ip.is_loopback,
            ip.is_link_local,
            ip.is_multicast,
            ip.is_reserved,
            ip.is_unspecified,
        )
    )

