"""Local review UI: show each image beside its OCR text, edit, and save.

Saving writes `corrected_text` and review_status="corrected" into the result
JSON, giving ground-truth pairs (image + corrected_text).
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from urllib.parse import unquote
import json
import webbrowser

from rag_ingest.stores import VisionStore

PAGE = files(__package__).joinpath("page.html").read_bytes()
IMAGE_TYPES = {".png": "image/png", ".webp": "image/webp"}


def make_handler(store: VisionStore):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, body: bytes, ctype: str, code: int = 200):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def not_found(self):
            self.send(b"not found", "text/plain", 404)

        def stem(self, prefix: str) -> str:
            return unquote(self.path[len(prefix):]).removesuffix(".json")

        def do_GET(self):
            if self.path == "/":
                self.send(PAGE, "text/html; charset=utf-8")
            elif self.path == "/api/items":
                items = []
                for stem in store.stems():
                    record = store.load(stem)
                    items.append({"name": f"{stem}.json", "text": record["text"],
                                  "corrected_text": record.get("corrected_text"),
                                  "status": record.get("review_status")})
                self.send(json.dumps(items, ensure_ascii=False).encode("utf-8"),
                          "application/json")
            elif self.path.startswith("/image/"):
                stem = self.stem("/image/")
                if not store.has(stem):
                    return self.not_found()
                image = store.image_path(store.load(stem))
                if not image.is_file():
                    return self.not_found()
                self.send(image.read_bytes(), IMAGE_TYPES.get(image.suffix.lower(), "image/jpeg"))
            else:
                self.not_found()

        def do_POST(self):
            if not self.path.startswith("/api/save/"):
                return self.not_found()
            stem = self.stem("/api/save/")
            if not store.has(stem):
                return self.not_found()
            length = int(self.headers.get("Content-Length", 0))
            store.save_correction(stem, self.rfile.read(length).decode("utf-8"))
            self.send(b"ok", "text/plain")

    return Handler


def serve(store: VisionStore, port: int = 8765, open_browser: bool = True) -> None:
    if not store.stems():
        raise FileNotFoundError(f"No result JSON files in {store.folder}")
    url = f"http://127.0.0.1:{port}/"
    print(f"Reviewing {store.folder}\nOpen {url}  (Ctrl+C to stop)")
    if open_browser:
        webbrowser.open(url)
    ThreadingHTTPServer(("127.0.0.1", port), make_handler(store)).serve_forever()
