"""Repositories for OCR results on disk. The file formats are defined here and nowhere else.

  VisionStore  outputs/<method>/<image>.json   one JSON record per image
  LineStore    outputs/<method>/<image>.txt    "text<TAB>confidence=0.1234" per line
"""
from pathlib import Path, PureWindowsPath
import json
import math

from rag_ingest.config import ENCODING, METHODS_BY_NAME, Paths
from rag_ingest.models import OcrLine, Recognition

_CONF_SEP = "\tconfidence="


class VisionStore:
    """JSON results from vision models, including reviewers' corrections."""
    kind = "vision"

    def __init__(self, folder: Path, paths: Paths):
        self.folder = folder
        self.paths = paths

    @property
    def name(self) -> str:
        return self.folder.name

    def path(self, stem: str) -> Path:
        return self.folder / f"{Path(stem).name}.json"  # .name: no path traversal

    def has(self, stem: str) -> bool:
        return self.path(stem).exists()

    def stems(self) -> list[str]:
        return [p.stem for p in sorted(self.folder.glob("*.json"))]

    def load(self, stem: str) -> dict:
        return json.loads(self.path(stem).read_text(encoding=ENCODING))

    def write(self, stem: str, record: dict) -> Path:
        self.folder.mkdir(parents=True, exist_ok=True)
        out = self.path(stem)
        out.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding=ENCODING)
        return out

    def save(self, image: Path, result: Recognition) -> Path:
        record = {
            "image": self.paths.relative(image),
            "model": result.model,
            "prompt_file": self.paths.relative(result.prompt_file or self.paths.prompt),
            "settings": result.settings,
            "text": result.text,
            "review_status": "pending",
        }
        return self.write(image.stem, record)

    def save_correction(self, stem: str, text: str) -> None:
        record = self.load(stem)
        record["corrected_text"] = text.strip()
        record["review_status"] = "corrected"
        self.write(stem, record)

    def image_path(self, record: dict) -> Path:
        # Records written on Windows store "outputs\\images\\x.png".
        stored = record["image"]
        if "\\" in stored:
            stored = PureWindowsPath(stored).as_posix()
        return self.paths.resolve(stored)

    def texts(self) -> dict[str, str]:
        return {stem: self.load(stem)["text"] for stem in self.stems()}

    def final_texts(self) -> dict[str, str]:
        """Best available text per image: a reviewer's correction wins over the model's."""
        out = {}
        for stem in self.stems():
            record = self.load(stem)
            out[stem] = (record.get("corrected_text") or record.get("text") or "").strip()
        return out

    def corrections(self) -> dict[str, str]:
        out = {}
        for stem in self.stems():
            corrected = self.load(stem).get("corrected_text")
            if corrected is not None:
                out[stem] = corrected
        return out


class LineStore:
    """Line-by-line results with confidence (EasyOCR, Tesseract)."""
    kind = "lines"

    def __init__(self, folder: Path):
        self.folder = folder

    @property
    def name(self) -> str:
        return self.folder.name

    def path(self, stem: str) -> Path:
        return self.folder / f"{Path(stem).name}.txt"

    def has(self, stem: str) -> bool:
        return self.path(stem).exists()

    def stems(self) -> list[str]:
        return [p.stem for p in sorted(self.folder.glob("*.txt"))]

    def lines(self, stem: str) -> list[OcrLine]:
        path = self.path(stem)
        if not path.is_file():
            return []
        out = []
        for raw in path.read_text(encoding=ENCODING).splitlines():
            text, sep, conf = raw.rpartition(_CONF_SEP)
            if not sep:
                out.append(OcrLine(raw, math.nan))
                continue
            try:
                out.append(OcrLine(text, float(conf)))
            except ValueError:
                out.append(OcrLine(text, math.nan))
        return out

    def confident_lines(self, stem: str, min_confidence: float) -> list[str]:
        # NaN (missing/bad confidence) never passes the threshold.
        return [l.text for l in self.lines(stem) if l.text and l.confidence >= min_confidence]

    def text(self, stem: str) -> str:
        return "\n".join(l.text for l in self.lines(stem))

    def save(self, image: Path, result: Recognition) -> Path:
        self.folder.mkdir(parents=True, exist_ok=True)
        out = self.path(image.stem)
        out.write_text("".join(f"{l.text}{_CONF_SEP}{l.confidence:.4f}\n"
                               for l in result.lines or []), encoding=ENCODING)
        return out

    def texts(self) -> dict[str, str]:
        return {stem: self.text(stem) for stem in self.stems()}

    def final_texts(self) -> dict[str, str]:
        return {stem: text.strip() for stem, text in self.texts().items()}


ResultStore = VisionStore | LineStore


def _folder(paths: Paths, folder: str | Path) -> Path:
    """A bare name ("vision_hint") means outputs/<name>; anything else is a path."""
    folder = Path(folder)
    if not folder.is_absolute() and len(folder.parts) == 1:
        return paths.method(folder.name)
    return paths.resolve(folder)


def _detect_kind(folder: Path) -> str:
    known = METHODS_BY_NAME.get(folder.name)
    if known:
        return known.kind
    only_text = any(folder.glob("*.txt")) and not any(folder.glob("*.json"))
    return "lines" if only_text else "vision"


def open_store(paths: Paths, folder: str | Path, kind: str | None = None) -> ResultStore:
    """Factory: the repository for a result folder, by known method or by its files."""
    folder = _folder(paths, folder)
    kind = kind or _detect_kind(folder)
    return VisionStore(folder, paths) if kind == "vision" else LineStore(folder)
