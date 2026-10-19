"""Local review UI: show each image beside its OCR text, edit, and save.

Saving writes `corrected_text` and review_status="corrected" into the result
JSON, giving ground-truth pairs (image + corrected_text).

Only the page this server rendered can save: requests must name this server
in Host (blocks DNS rebinding), and saves need the per-run token embedded in
the page (a custom header, which other sites cannot send without a CORS
preflight that is never approved).

The server listens on the loopback interface only, so plain HTTP never leaves
this machine; TLS would need a self-signed certificate the browser rejects.
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from urllib.parse import unquote
import json
import secrets
import webbrowser

from rag_ingest.config import ENCODING
from rag_ingest.stores import VisionStore

LOOPBACK = "127.0.0.1"
PAGE = files(__package__).joinpath("page.html").read_text(encoding=ENCODING)
TEXT_PLAIN = "text/plain"
HTML = "text/html; charset=utf-8"
JSON = "application/json"
IMAGE_TYPES = {".png": "image/png", ".webp": "image/webp", ".gif": "image/gif",
               ".bmp": "image/bmp"}


def items_json(store: VisionStore) -> bytes:
    items = []
    for stem in store.stems():
        record = store.load(stem)
        items.append({"name": f"{stem}.json", "text": record["text"],
                      "corrected_text": record.get("corrected_text"),
                      "status": record.get("review_status")})
    return json.dumps(items, ensure_ascii=False).encode(ENCODING)


def image_file(store: VisionStore, stem: str) -> Path | None:
    """The image behind one result, or None if the result or the file is missing."""
    if not store.has(stem):
        return None
    image = store.image_path(store.load(stem))
    return image if image.is_file() else None


class ReviewHandler(BaseHTTPRequestHandler):
    """Routes requests to one VisionStore. Configure with make_handler()."""
    store: VisionStore
    hosts: frozenset[str]
    origins: frozenset[str]
    token: str
    page: bytes

    def log_message(self, *_args):
        """Silence the default per-request access log; the terminal shows only the URL."""

    def send(self, body: bytes, ctype: str, code: int = 200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def not_found(self):
        self.send(b"not found", TEXT_PLAIN, 404)

    def forbidden(self):
        self.send(b"forbidden", TEXT_PLAIN, 403)

    def same_host(self) -> bool:
        return self.headers.get("Host") in self.hosts

    def from_review_page(self) -> bool:
        origin = self.headers.get("Origin")
        token = self.headers.get("X-Review-Token", "")
        return ((origin is None or origin in self.origins)
                and secrets.compare_digest(token, self.token))

    def stem(self, prefix: str) -> str:
        return unquote(self.path[len(prefix):]).removesuffix(".json")

    def do_GET(self):
        if not self.same_host():
            return self.forbidden()
        if self.path == "/":
            return self.send(self.page, HTML)
        if self.path == "/api/items":
            return self.send(items_json(self.store), JSON)
        if self.path.startswith("/image/"):
            return self.send_image(self.stem("/image/"))
        self.not_found()

    def send_image(self, stem: str):
        image = image_file(self.store, stem)
        if image is None:
            return self.not_found()
        self.send(image.read_bytes(), IMAGE_TYPES.get(image.suffix.lower(), "image/jpeg"))

    def do_POST(self):
        if not (self.same_host() and self.from_review_page()):
            return self.forbidden()
        if not self.path.startswith("/api/save/"):
            return self.not_found()
        stem = self.stem("/api/save/")
        if not self.store.has(stem):
            return self.not_found()
        length = int(self.headers.get("Content-Length", 0))
        self.store.save_correction(stem, self.rfile.read(length).decode(ENCODING))
        self.send(b"ok", TEXT_PLAIN)


def make_handler(store: VisionStore, port: int, token: str) -> type[ReviewHandler]:
    hosts = frozenset({f"{LOOPBACK}:{port}", f"localhost:{port}"})
    return type("BoundReviewHandler", (ReviewHandler,), {
        "store": store,
        "hosts": hosts,
        "origins": frozenset(f"http://{h}" for h in hosts),  # NOSONAR loopback only, see module doc
        "token": token,
        "page": PAGE.replace("__TOKEN__", token).encode(ENCODING),
    })


def serve(store: VisionStore, port: int = 8765, open_browser: bool = True) -> None:
    if not store.stems():
        raise FileNotFoundError(f"No result JSON files in {store.folder}")
    url = f"http://{LOOPBACK}:{port}/"  # NOSONAR loopback only, see module doc
    print(f"Reviewing {store.folder}\nOpen {url}  (Ctrl+C to stop)")
    if open_browser:
        webbrowser.open(url)
    handler = make_handler(store, port, secrets.token_urlsafe(32))
    ThreadingHTTPServer((LOOPBACK, port), handler).serve_forever()  # NOSONAR loopback only
