"""Ground truth for the OCR benchmark: what each screenshot really says.

outputs/ground_truth/<image>.txt   the transcription (plain text, reading order)
outputs/ground_truth/provenance.json
    <image> -> {"by": "human" | "claude", "method": ..., "date": ..., "note": ...}

Who made a transcription, and how, matters for how far the benchmark can be
trusted: a correction edited from one model's output leans towards that
model, and a transcription made by an AI model is not a human check. The
benchmark therefore reports its numbers per provenance.

Methods:
blank box    typed from the image alone, no model output shown (`rag-ingest
            review --blank`), so no engine is favoured
edited from <folder>   a model's output corrected in `rag-ingest review`
"""
from dataclasses import dataclass
from datetime import date
from pathlib import Path
import json

from rag_ingest.config import ENCODING

HUMAN, CLAUDE = "human", "claude"
BLANK_BOX = "blank box"
PROVENANCE_FILE = "provenance.json"
# ingest manifest statuses of images that are benchmarked (icons, duplicates and
# undecodable images are not: icons hold no text, duplicates would count twice)
BENCHMARK_STATUSES = ("ocr", "needs_ocr")


def edited_from(folder: str) -> str:
    return f"edited from {folder} model text"


@dataclass
class GroundTruthStore:
    folder: Path

    def path(self, stem: str) -> Path:
        return self.folder / f"{Path(stem).name}.txt"  # .name: no path traversal

    def has(self, stem: str) -> bool:
        return self.path(stem).exists()

    def texts(self) -> dict[str, str]:
        return {p.stem: p.read_text(encoding=ENCODING) for p in sorted(self.folder.glob("*.txt"))}

    def provenance(self) -> dict[str, dict]:
        path = self.folder / PROVENANCE_FILE
        return json.loads(path.read_text(encoding=ENCODING)) if path.exists() else {}

    def save(self, stem: str, text: str, by: str, method: str, note: str = "") -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        self.path(stem).write_text(text.strip() + "\n", encoding=ENCODING)
        provenance = self.provenance()
        provenance[Path(stem).name] = {"by": by, "method": method, "date": date.today().isoformat(),
                                       **({"note": note} if note else {})}
        (self.folder / PROVENANCE_FILE).write_text(
            json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True), encoding=ENCODING)


def benchmark_images(ingest_manifest: Path) -> list[str] | None:
    """Images worth a ground truth: unique and holding text, per the ingest manifest
    (None when there is no manifest; then every image counts)."""
    if not ingest_manifest.exists():
        return None
    entries = json.loads(ingest_manifest.read_text(encoding=ENCODING))
    return sorted(e["image"] for e in entries if e.get("status") in BENCHMARK_STATUSES)
