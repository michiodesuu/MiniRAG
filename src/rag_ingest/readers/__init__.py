"""Source documents: pick a reader by file extension (Strategy, chosen from a registry).

Every reader turns a file into the same Page objects, so everything after
this point (OCR lookup, quality rules, merging) is format-independent.
Readers import their libraries only when a file of that type is read.
"""
from collections import Counter
from importlib import import_module
from pathlib import Path
from typing import Iterator, Protocol
import sys

from rag_ingest.models import Page


class DocumentReader(Protocol):
    def __call__(self, path: Path) -> Iterator[Page]: ...


# suffix -> "module:function" implementing DocumentReader, resolved lazily.
READERS = {
    ".pdf": "rag_ingest.readers.pdf:read_pages",
    ".docx": "rag_ingest.readers.word:read_pages",
}


def supported(path: Path) -> bool:
    return path.suffix.lower() in READERS


def reader_for(path: Path) -> DocumentReader:
    try:
        module, function = READERS[path.suffix.lower()].split(":")
    except KeyError:
        raise ValueError(f"Unsupported document type {path.suffix!r}: {path.name}; "
                        f"supported: {', '.join(READERS)}") from None
    return getattr(import_module(module), function)


def read_pages(path: Path) -> Iterator[Page]:
    return reader_for(path)(path)


IGNORED_NAMES = (".gitkeep", ".DS_Store", "Thumbs.db", "desktop.ini")


def _is_candidate(path: Path) -> bool:
    """A real file: not a placeholder, OS metadata, hidden, or a Word lock file (~$x.docx)."""
    return (path.is_file() and path.name not in IGNORED_NAMES
            and not path.name.startswith((".", "~$")))


def list_documents(folder: Path) -> list[Path]:
    """Supported files in a folder and its sub-folders (e.g. data/raw/pdf/, data/raw/docx/),
    sorted by path. Files no reader handles yet are reported on stderr, not read."""
    files = sorted(p for p in folder.rglob("*") if _is_candidate(p))
    unsupported = [p for p in files if not supported(p)]
    if unsupported:
        names = ", ".join(str(p.relative_to(folder)) for p in unsupported)
        print(f"Skipping {len(unsupported)} file(s) with no reader yet: {names}", file=sys.stderr)
    docs = [p for p in files if supported(p)]
    if not docs:
        raise FileNotFoundError(f"No documents ({', '.join(READERS)}) found in {folder}")
    clashes = [stem for stem, n in Counter(p.stem for p in docs).items() if n > 1]
    if clashes:
        # Image results are named <stem>_page_N_image_M, so pdf/a.pdf and docx/a.docx collide.
        raise ValueError(f"Documents share a name, rename one of each: {', '.join(clashes)}")
    return docs
