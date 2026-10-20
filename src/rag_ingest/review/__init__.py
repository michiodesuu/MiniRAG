"""Local review UI: an image beside a text box; edit and save.

Two sources of items (Strategy):
  CorrectionSource  a model's result folder: the box starts with the model's
                    text; saving writes `corrected_text` into its JSON
  BlankSource       blank box: the box starts empty (or with your own earlier
                    blank-box text), no model output is shown, so no engine is
                    favoured; saving writes outputs/ground_truth/<image>.txt
                    with provenance "human, blank box". Only images worth a
                    ground truth are listed (no icons, no duplicates).

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
from typing import Protocol
import json
import secrets
import webbrowser

from rag_ingest.config import ENCODING
from rag_ingest.ground_truth import BLANK_BOX, HUMAN, GroundTruthStore
from rag_ingest.stores import VisionStore

LOOPBACK = "127.0.0.1"
PAGE = files(__package__).joinpath("page.html").read_text(encoding=ENCODING)
TEXT_PLAIN = "text/plain"
HTML = "text/html; charset=utf-8"
JSON = "application/json"
IMAGE_TYPES = {".png": "image/png", ".webp": "image/webp", ".gif": "image/gif",
               ".bmp": "image/bmp"}


class ReviewSource(Protocol):
    def items(self) -> list[dict]: ...           # {"name", "text", "corrected_text", "status"}

    def image(self, stem: str) -> Path | None: ...

    def has(self, stem: str) -> bool: ...

    def save(self, stem: str, text: str) -> None: ...

    def describe(self) -> str: ...


class CorrectionSource:
    def __init__(self, store: VisionStore):
        self.store = store

    def items(self) -> list[dict]:
        out = []
        for stem in self.store.stems():
            record = self.store.load(stem)
            out.append({"name": f"{stem}.json", "text": record["text"],
                        "corrected_text": record.get("corrected_text"),
                        "status": record.get("review_status")})
        return out

    def image(self, stem: str) -> Path | None:
        if not self.store.has(stem):
            return None
        image = self.store.image_path(self.store.load(stem))
        return image if image.is_file() else None

    def has(self, stem: str) -> bool:
        return self.store.has(stem)

    def save(self, stem: str, text: str) -> None:
        self.store.save_correction(stem, text)

    def describe(self) -> str:
        return f"correcting {self.store.folder}"


class BlankSource:
    def __init__(self, truth: GroundTruthStore, images: dict[str, Path]):
        self.truth, self.images = truth, images  # stem -> image file

    def items(self) -> list[dict]:
        provenance, out = self.truth.provenance(), []
        for stem in self.images:
            mine = provenance.get(stem, {}).get("by") == HUMAN and self.truth.has(stem)
            text = self.truth.path(stem).read_text(encoding=ENCODING).strip() if mine else None
            out.append({"name": stem, "text": "", "corrected_text": text,
                        "status": "corrected" if mine else None})
        return out

    def image(self, stem: str) -> Path | None:
        path = self.images.get(stem)
        return path if path and path.is_file() else None

    def has(self, stem: str) -> bool:
        return stem in self.images

    def save(self, stem: str, text: str) -> None:
        self.truth.save(stem, text, HUMAN, BLANK_BOX)

    def describe(self) -> str:
        return f"blank box -> {self.truth.folder} ({len(self.images)} images)"


class ReviewHandler(BaseHTTPRequestHandler):
    """Routes requests to one ReviewSource. Configure with make_handler()."""
    source: ReviewSource
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
            return self.send(json.dumps(self.source.items(), ensure_ascii=False).encode(ENCODING), JSON)
        if self.path.startswith("/image/"):
            return self.send_image(self.stem("/image/"))
        self.not_found()

    def send_image(self, stem: str):
        image = self.source.image(stem)
        if image is None:
            return self.not_found()
        self.send(image.read_bytes(), IMAGE_TYPES.get(image.suffix.lower(), "image/jpeg"))

    def do_POST(self):
        if not (self.same_host() and self.from_review_page()):
            return self.forbidden()
        if not self.path.startswith("/api/save/"):
            return self.not_found()
        stem = self.stem("/api/save/")
        if not self.source.has(stem):
            return self.not_found()
        length = int(self.headers.get("Content-Length", 0))
        self.source.save(stem, self.rfile.read(length).decode(ENCODING))
        self.send(b"ok", TEXT_PLAIN)


def make_handler(source: ReviewSource | VisionStore, port: int, token: str) -> type[ReviewHandler]:
    if isinstance(source, VisionStore):
        source = CorrectionSource(source)
    hosts = frozenset({f"{LOOPBACK}:{port}", f"localhost:{port}"})
    return type("BoundReviewHandler", (ReviewHandler,), {
        "source": source,
        "hosts": hosts,
        "origins": frozenset(f"http://{h}" for h in hosts),  # NOSONAR loopback only, see module doc
        "token": token,
        "page": PAGE.replace("__TOKEN__", token).encode(ENCODING),
    })


def serve(source: ReviewSource, port: int = 8765, open_browser: bool = True) -> None:
    if not source.items():
        raise FileNotFoundError(f"Nothing to review ({source.describe()})")
    url = f"http://{LOOPBACK}:{port}/"  # NOSONAR loopback only, see module doc
    print(f"Reviewing: {source.describe()}\nOpen {url}  (Ctrl+C to stop)")
    if open_browser:
        webbrowser.open(url)
    handler = make_handler(source, port, secrets.token_urlsafe(32))
    ThreadingHTTPServer((LOOPBACK, port), handler).serve_forever()  # NOSONAR loopback only
