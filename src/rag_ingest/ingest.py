"""Build RAG-ready text per PDF: text layer + screenshot OCR, in reading order.

Pipeline, per page:
  1. text blocks from the PDF's own text layer (no OCR)
  2. each embedded image: skip icons, dedupe by md5, take OCR text from the
     first result folder whose text passes the quality rules (cached by md5)
  3. merge blocks and screenshot text by position on the page

No model runs here, so memory stays small.
"""
from dataclasses import dataclass, field
from pathlib import Path
import json

from rag_ingest.pdf import EmbeddedImage, Page, list_pdfs, read_pages
from rag_ingest.quality import rejection
from rag_ingest.stores import ResultStore

DEFAULT_SOURCES = ("vision_typhoon", "vision_hint")
ICON_MAX_SIDE = 120


class OcrCache:
    """md5 -> {"text", "source", "image"}; only usable text is kept."""

    def __init__(self, path: Path):
        self.path = path
        stored = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        self.entries = {k: v for k, v in stored.items() if rejection(v["text"]) is None}

    def get(self, digest: str) -> dict | None:
        return self.entries.get(digest)

    def put(self, digest: str, text: str, source: str, image: str) -> None:
        self.entries[digest] = {"text": text, "source": source, "image": image}

    def save(self) -> None:
        self.path.write_text(json.dumps(self.entries, ensure_ascii=False, indent=2),
                             encoding="utf-8")


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

    def screenshot_text(self, image: EmbeddedImage) -> str | None:
        """Text to insert for one image (None = nothing), recording it in the manifest."""
        stem, digest = image.ref.stem, image.md5
        entry = {"image": stem, "pdf": f"{image.ref.document}.pdf", "page": image.ref.page,
                 "size": [image.width, image.height], "md5": digest}
        self.result.manifest.append(entry)

        if image.short_side < self.min_side:
            entry["status"] = "icon"
            return None
        if digest in self.result.unique:
            entry["status"] = "duplicate"
            entry["duplicate_of"] = self.result.unique[digest]
        else:
            self.result.unique[digest] = stem

        if self.cache.get(digest) is None:
            text, source, rejected = self.sources.lookup(stem)
            if rejected:
                entry["rejected"] = rejected
            if text is not None:
                self.cache.put(digest, text, source, stem)

        cached = self.cache.get(digest)
        if cached:
            entry.setdefault("status", "ocr")
            entry["source"] = cached["source"]
            return f"[ภาพหน้าจอ]\n{cached['text']}\n[/ภาพหน้าจอ]"
        entry.setdefault("status", "needs_ocr")
        if "duplicate_of" not in entry:
            self.result.needs_ocr.append(stem)
        return f"[ภาพหน้าจอ: ยังไม่ได้ OCR ({stem})]"

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

    def document_text(self, pdf: Path) -> str:
        return "\n\n".join(self.page_text(page) for page in read_pages(pdf)) + "\n"


def run_ingest(pdf_dir: Path, out_dir: Path, stores: list[ResultStore],
               min_side: int = ICON_MAX_SIDE) -> IngestResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = OcrCache(out_dir / "ocr_cache.json")
    ingestor = Ingestor(OcrSourceChain(stores), cache, min_side)

    for pdf in list_pdfs(pdf_dir):
        out = out_dir / f"{pdf.stem}.md"
        out.write_text(ingestor.document_text(pdf), encoding="utf-8")
        print(f"{pdf.name} -> {out}")

    result = ingestor.result
    cache.save()
    (out_dir / "manifest.json").write_text(
        json.dumps(result.manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "needs_ocr.txt").write_text("".join(f"{n}\n" for n in result.needs_ocr),
                                           encoding="utf-8")
    return result
