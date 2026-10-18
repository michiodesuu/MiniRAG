"""Run one engine over many images: skip-existing, per-image failure isolation, summary."""
from dataclasses import dataclass
from pathlib import Path

from rag_ingest.engines import OcrEngine
from rag_ingest.stores import ResultStore


@dataclass
class BatchSummary:
    done: int = 0
    skipped: int = 0
    failed: int = 0

    def __str__(self) -> str:
        return f"Done: {self.done} | Skipped: {self.skipped} | Failed: {self.failed}"


def run_batch(engine: OcrEngine, store: ResultStore, images: list[Path],
              skip_existing: bool = False) -> BatchSummary:
    summary = BatchSummary()
    for i, image in enumerate(images, 1):
        label = f"[{i}/{len(images)}] {image.name}"
        if skip_existing and store.has(image.stem):
            print(f"{label} - skipped (exists)", flush=True)
            summary.skipped += 1
            continue
        try:
            result = engine.recognize(image)
            store.save(image, result)
        except Exception as error:  # one bad image must not stop a long CPU run
            print(f"{label} - FAILED: {error}", flush=True)
            summary.failed += 1
            continue
        detail = f" ({len(result.lines)} lines)" if result.lines is not None else ""
        print(f"{label} - ok{detail}", flush=True)
        summary.done += 1
    return summary
