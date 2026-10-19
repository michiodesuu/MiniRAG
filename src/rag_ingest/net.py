"""Network helpers shared by every client of an internal service (OCR API, embedding API)."""
import sys
from urllib.parse import urlsplit

LOOPBACK_HOSTS = ("localhost", "127.0.0.1", "::1")


def warn_if_clear_text(url: str, what: str = "images") -> None:
    """Warn when data (and any API key) would cross the network unencrypted."""
    parts = urlsplit(url)
    if parts.scheme != "https" and parts.hostname not in LOOPBACK_HOSTS:
        print(f"WARNING: {parts.scheme}:// sends {what} unencrypted to {parts.hostname}; "
              "use https if the server supports it.", file=sys.stderr)
