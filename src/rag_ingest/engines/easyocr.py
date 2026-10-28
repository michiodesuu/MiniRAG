"""EasyOCR (Thai + English) line OCR. Heavy: pulls in PyTorch and downloads models on first use."""
from pathlib import Path

from rag_ingest.imaging import load_rgb
from rag_ingest.models import OcrLine, Recognition


class EasyOcrEngine:
    store_kind = "lines"
    default_output = "easyocr"

    def __init__(self, langs: tuple[str, ...] = ("th", "en"), upscale: int = 0):
        import easyocr

        self.langs, self.upscale = list(langs), upscale
        self._reader = easyocr.Reader(self.langs)

    def recognize(self, image: Path) -> Recognition:
        import numpy as np

        # paragraph=False keeps one confidence per box; detail=1 returns (box, text, score).
        found = self._reader.readtext(np.asarray(load_rgb(image, self.upscale)))
        lines = [OcrLine(text, float(score)) for _, text, score in found if text.strip()]
        return Recognition(
            text="\n".join(l.text for l in lines),
            lines=lines,
            model="easyocr",
            settings={"lang": "+".join(self.langs), "upscale_min_side": self.upscale},
        )
