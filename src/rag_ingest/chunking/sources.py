"""The ingest output as chunking reads it: Markdown files, manifest.json, ocr_cache.json.

A screenshot block in the Markdown carries only its OCR text. Its image (md5)
and OCR source are found by matching that text against the OCR cache among
the images the manifest lists on the same page of the same document.
"""
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json
import unicodedata

from rag_ingest.config import ENCODING

SCREENSHOT_STATUSES = ("ocr", "duplicate")


def nfc_name(name: str) -> str:
    """A file name as compared for exclusion: one Unicode form (macOS can return
    decomposed names), invisible format characters removed, case folded."""
    name = unicodedata.normalize("NFC", name)
    return "".join(ch for ch in name if unicodedata.category(ch) != "Cf").strip().casefold()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class DocumentImages:
    """Screenshots of one document: page -> [(ocr text, md5, source)]."""
    by_page: dict[int, list[tuple[str, str, str | None]]] = field(default_factory=dict)
    source_types: set[str] = field(default_factory=set)  # every suffix the manifest shows
    md5s: set[str] = field(default_factory=set)          # every embedded image, any status

    @property
    def doc_type(self) -> str | None:
        """The source type, when the manifest agrees on exactly one."""
        return next(iter(self.source_types)) if len(self.source_types) == 1 else None

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
            if nfc_name(Path(source_file).stem) == nfc_name(document_stem):
                self._add(images, entry, source_file)
        return images

    def _add(self, images: DocumentImages, entry: dict, source_file: str) -> None:
        images.source_types.add(Path(source_file).suffix.lstrip(".").lower() or "unknown")
        if entry.get("md5"):
            images.md5s.add(entry["md5"])
        cached = self.cache.get(entry.get("md5", ""))
        if entry.get("status") in SCREENSHOT_STATUSES and cached:
            images.by_page.setdefault(entry["page"], []).append(
                (cached["text"], entry["md5"], entry.get("source")))


def find_doc_type(stem: str, images: DocumentImages, documents_dir: Path) -> str:
    """pdf / docx from the manifest, else from the source file in data/raw, else "unknown".
    For metadata only: exclusion never relies on the data/raw lookup."""
    if images.doc_type:
        return images.doc_type
    found = sorted(documents_dir.rglob(f"{stem}.*")) if documents_dir.exists() else []
    return found[0].suffix.lstrip(".").lower() if found else "unknown"
