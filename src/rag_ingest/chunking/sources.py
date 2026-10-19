"""The ingest output as chunking reads it: Markdown files, manifest.json, ocr_cache.json.

A screenshot block in the Markdown carries only its OCR text. Its image (md5)
and OCR source are found by matching that text against the OCR cache among
the images the manifest lists on the same page of the same document.
"""
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json

from rag_ingest.config import ENCODING

SCREENSHOT_STATUSES = ("ocr", "duplicate")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class DocumentImages:
    """Screenshots of one document: page -> [(ocr text, md5, source)]."""
    by_page: dict[int, list[tuple[str, str, str | None]]] = field(default_factory=dict)
    doc_type: str | None = None

    def match(self, page: int, text: str) -> tuple[str | None, str | None]:
        """(md5, ocr source) of the screenshot on this page with exactly this text."""
        return next(((md5, source) for t, md5, source in self.by_page.get(page, []) if t == text),
                    (None, None))


@dataclass
class IngestOutput:
    folder: Path
    manifest: list[dict]
    cache: dict[str, dict]

    @classmethod
    def load(cls, folder: Path) -> "IngestOutput":
        def read(name: str, default):
            path = folder / name
            return json.loads(path.read_text(encoding=ENCODING)) if path.exists() else default
        return cls(folder, read("manifest.json", []), read("ocr_cache.json", {}))

    def markdown_files(self) -> list[Path]:
        return sorted(self.folder.glob("*.md"))

    def images(self, document_stem: str) -> DocumentImages:
        images = DocumentImages()
        for entry in self.manifest:
            source_file = entry.get("document") or entry.get("pdf") or ""  # "pdf": older manifests
            if Path(source_file).stem != document_stem:
                continue
            images.doc_type = Path(source_file).suffix.lstrip(".").lower() or None
            cached = self.cache.get(entry.get("md5", ""))
            if entry.get("status") in SCREENSHOT_STATUSES and cached:
                images.by_page.setdefault(entry["page"], []).append(
                    (cached["text"], entry["md5"], entry.get("source")))
        return images


def find_doc_type(stem: str, images: DocumentImages, documents_dir: Path) -> str:
    """pdf / docx from the manifest, else from the source file in data/raw, else "unknown"."""
    if images.doc_type:
        return images.doc_type
    found = sorted(documents_dir.rglob(f"{stem}.*")) if documents_dir.exists() else []
    return found[0].suffix.lstrip(".").lower() if found else "unknown"
