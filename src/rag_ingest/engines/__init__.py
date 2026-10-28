"""OCR engines behind one interface (Strategy), created by name (Factory).

Engines import their heavy libraries only when constructed, so a run that
uses Tesseract never loads EasyOCR or the Ollama client into memory.
"""
from importlib import import_module
from pathlib import Path
from typing import Protocol

from rag_ingest.models import Recognition


class OcrEngine(Protocol):
    store_kind: str      # "vision" -> JSON results, "lines" -> text<TAB>confidence
    default_output: str  # result folder under outputs/

    def recognize(self, image: Path) -> Recognition: ...


# name -> "module:factory"; resolved lazily by create_engine.
ENGINES = {
    "tesseract": "rag_ingest.engines.tesseract:TesseractEngine",
    "easyocr": "rag_ingest.engines.easyocr:EasyOcrEngine",
    "ollama": "rag_ingest.engines.vision:ollama_engine",
    "api": "rag_ingest.engines.vision:api_engine",
}


def create_engine(name: str, **options) -> OcrEngine:
    try:
        module, factory = ENGINES[name].split(":")
    except KeyError:
        raise ValueError(f"Unknown engine {name!r}; choose from {', '.join(ENGINES)}") from None
    return getattr(import_module(module), factory)(**options)
