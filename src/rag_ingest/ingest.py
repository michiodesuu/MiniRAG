"""Build RAG-ready text per document (PDF or Word): its own text + screenshot OCR, in reading order.

Pipeline, per page:
  1. text blocks from the document's own text (PDF text layer, Word paragraphs/tables; no OCR)
  2. each embedded image: skip icons, dedupe by md5, take OCR text from the
     first result folder whose text passes the quality rules (cached by md5)
  3. merge blocks and screenshot text by position on the page

No model runs here, so memory stays small.
"""
from dataclasses import dataclass, field
from pathlib import Path
import json

from rag_ingest.config import ENCODING
from rag_ingest.readers import list_documents, read_pages
from rag_ingest.models import EmbeddedImage, Page
from rag_ingest.quality import rejection
from rag_ingest.stores import ResultStore

DEFAULT_SOURCES = ("vision_typhoon", "vision_hint")
ICON_MAX_SIDE = 120
SCREENSHOT = "[ภาพหน้าจอ]\n{text}\n[/ภาพหน้าจอ]"
MISSING_OCR = "[ภาพหน้าจอ: ยังไม่ได้ OCR ({stem})]"


class OcrCache:
    """md5 -> {"text", "source", "image"}; only usable text is kept."""

    def __init__(self, path: Path):
        self.path = path
        stored = json.loads(path.read_text(encoding=ENCODING)) if path.exists() else {}
        self.entries = {k: v for k, v in stored.items() if rejection(v["text"]) is None}

    def get(self, digest: str) -> dict | None:
        return self.entries.get(digest)

    def put(self, digest: str, text: str, source: str, image: str) -> None:
        self.entries[digest] = {"text": text, "source": source, "image": image}

    def save(self) -> None:
        self.path.write_text(json.dumps(self.entries, ensure_ascii=False, indent=2),
                             encoding=ENCODING)


class OcrSourceChain:
    """Result folders in priority order (Chain of Responsibility): first usable text wins."""

    def __init__(self, stores: list[ResultStore]):
        self.sources = [(store.name, store.final_texts()) for store in stores
                        if store.folder.exists()]

    def lookup(self, stem: str) -> tuple[str | None, str | None, list[dict]]:
        """(text, source, rejections) for one image."""
        rejected = []
        for source, texts in self.sources:
            if stem not in texts:
                continue
            reason = rejection(texts[stem])
            if reason is None:
                return texts[stem], source, rejected
            rejected.append({"source": source, "reason": reason})
        return None, None, rejected


@dataclass
class IngestResult:
    manifest: list[dict] = field(default_factory=list)
    needs_ocr: list[str] = field(default_factory=list)
    unique: dict[str, str] = field(default_factory=dict)  # md5 -> first image stem

    def count(self, status: str) -> int:
        return sum(e["status"] == status for e in self.manifest)


class Ingestor:
    def __init__(self, sources: OcrSourceChain, cache: OcrCache, min_side: int = ICON_MAX_SIDE):
        self.sources, self.cache, self.min_side = sources, cache, min_side
        self.result = IngestResult()
        self.document = ""  # file name of the document being read

    def _entry(self, image: EmbeddedImage) -> dict:
        entry = {"image": image.ref.stem, "document": self.document, "page": image.ref.page,
                 "size": [image.width, image.height], "md5": image.md5}
        self.result.manifest.append(entry)
        return entry

    def _skip_status(self, image: EmbeddedImage) -> str | None:
        if not image.decodable:  # e.g. EMF/WMF drawings from Word: no OCR engine reads them
            return "unsupported_format"
        if image.short_side < self.min_side:
            return "icon"
        return None

    def _track_duplicate(self, entry: dict, digest: str, stem: str) -> None:
        first = self.result.unique.setdefault(digest, stem)
        if first != stem:
            entry["status"] = "duplicate"
            entry["duplicate_of"] = first

    def _ocr(self, entry: dict, digest: str, stem: str) -> dict | None:
        """Cached OCR for this image's bytes, looking it up in the sources once."""
        if self.cache.get(digest) is None:
            text, source, rejected = self.sources.lookup(stem)
            if rejected:
                entry["rejected"] = rejected
            if text is not None:
                self.cache.put(digest, text, source, stem)
        return self.cache.get(digest)

    def screenshot_text(self, image: EmbeddedImage) -> str | None:
        """Text to insert for one image (None = nothing), recording it in the manifest."""
        entry, stem = self._entry(image), image.ref.stem
        if status := self._skip_status(image):
            entry["status"] = status
            return None
        self._track_duplicate(entry, image.md5, stem)
        cached = self._ocr(entry, image.md5, stem)
        if cached:
            entry.setdefault("status", "ocr")
            entry["source"] = cached["source"]
            return SCREENSHOT.format(text=cached["text"])
        entry.setdefault("status", "needs_ocr")
        if "duplicate_of" not in entry:
            self.result.needs_ocr.append(stem)
        return MISSING_OCR.format(stem=stem)

    def page_text(self, page: Page) -> str:
        items = [(b.y, b.x, b.text) for b in page.blocks]
        for image in page.images:
            text = self.screenshot_text(image)
            if text is not None:
                # Images not drawn on the page go to the end.
                y, x = (image.y, image.x) if image.y is not None else (page.height, 0)
                items.append((y, x, text))
        items.sort(key=lambda item: (round(item[0]), item[1]))
        return f"<!-- page {page.number} -->\n" + "\n\n".join(text for *_, text in items)

    def document_text(self, document: Path) -> str:
        self.document = document.name
        return "\n\n".join(self.page_text(page) for page in read_pages(document)) + "\n"


def run_ingest(docs_dir: Path, out_dir: Path, stores: list[ResultStore],
               min_side: int = ICON_MAX_SIDE) -> IngestResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = OcrCache(out_dir / "ocr_cache.json")
    ingestor = Ingestor(OcrSourceChain(stores), cache, min_side)

    for document in list_documents(docs_dir):
        out = out_dir / f"{document.stem}.md"
        out.write_text(ingestor.document_text(document), encoding=ENCODING)
        print(f"{document.name} -> {out}")

    result = ingestor.result
    cache.save()
    (out_dir / "manifest.json").write_text(
        json.dumps(result.manifest, ensure_ascii=False, indent=2), encoding=ENCODING)
    (out_dir / "needs_ocr.txt").write_text("".join(f"{n}\n" for n in result.needs_ocr),
                                           encoding=ENCODING)
    return result
