"""Project layout: every folder the tools read or write, in one place."""
from dataclasses import dataclass
from pathlib import Path
import os

SUPPORTED_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp"})


def _default_root() -> Path:
    env = os.environ.get("RAG_INGEST_ROOT")
    return Path(env) if env else Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    root: Path

    @classmethod
    def default(cls) -> "Paths":
        return cls(_default_root())

    def resolve(self, path: str | Path) -> Path:
        """Relative paths are taken from the project root, not the current directory."""
        path = Path(path)
        return path if path.is_absolute() else self.root / path

    def relative(self, path: Path) -> str:
        """Portable (forward-slash) path for storing in result files."""
        return (path.relative_to(self.root) if path.is_relative_to(self.root) else path).as_posix()

    @property
    def pdfs(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def prompt(self) -> Path:
        return self.root / "prompts" / "thai_document_ocr.txt"

    @property
    def outputs(self) -> Path:
        return self.root / "outputs"

    @property
    def images(self) -> Path:
        return self.outputs / "images"

    @property
    def ingest(self) -> Path:
        return self.outputs / "ingest"

    @property
    def ground_truth(self) -> Path:
        return self.outputs / "ground_truth"

    def method(self, name: str) -> Path:
        """Result folder of one OCR method, e.g. method("vision_hint")."""
        return self.outputs / name


@dataclass(frozen=True)
class Method:
    """An OCR run whose results live in outputs/<name>/."""
    name: str
    kind: str  # "vision" (JSON per image) or "lines" (text<TAB>confidence per image)


# Known result folders, in report column order.
METHODS = (
    Method("vision", "vision"),
    Method("vision_hint", "vision"),
    Method("vision_both", "vision"),
    Method("vision_typhoon", "vision"),
    Method("ocr", "lines"),        # PaddleOCR
    Method("tesseract", "lines"),
)
METHODS_BY_NAME = {m.name: m for m in METHODS}

# Default places the vision engine takes hint lines from (--hint-source).
HINT_FOLDERS = {"paddle": "ocr", "tesseract": "tesseract"}
