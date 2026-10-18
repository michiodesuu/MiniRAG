"""Tesseract (Thai + English) line OCR."""
from pathlib import Path

from rag_ingest.imaging import load_rgb
from rag_ingest.models import OcrLine, Recognition
from rag_ingest.textproc import join_thai_words

# User-level winget install on Windows; elsewhere tesseract is expected on PATH.
WINDOWS_TESSERACT = Path.home() / "AppData/Local/Programs/Tesseract-OCR/tesseract.exe"


class TesseractEngine:
    store_kind = "lines"
    default_output = "tesseract"

    def __init__(self, psm: int = 3, upscale: int = 0, lang: str = "tha+eng"):
        import pytesseract

        if WINDOWS_TESSERACT.exists():
            pytesseract.pytesseract.tesseract_cmd = str(WINDOWS_TESSERACT)
        self._tesseract = pytesseract
        self.psm, self.upscale, self.lang = psm, upscale, lang

    def recognize(self, image: Path) -> Recognition:
        data = self._tesseract.image_to_data(
            load_rgb(image, self.upscale),
            lang=self.lang,
            config=f"--psm {self.psm}",
            output_type=self._tesseract.Output.DICT,
        )

        # Group words into lines; a line's confidence is its words' mean.
        grouped: dict[tuple, list[tuple[str, float]]] = {}
        for i, word in enumerate(data["text"]):
            conf = float(data["conf"][i])
            if not word.strip() or conf < 0:
                continue
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            grouped.setdefault(key, []).append((word.strip(), conf))

        lines = [
            OcrLine(join_thai_words(" ".join(w for w, _ in words)),
                    sum(c for _, c in words) / len(words) / 100)
            for words in grouped.values()
        ]
        return Recognition(
            text="\n".join(l.text for l in lines),
            lines=lines,
            model="tesseract",
            settings={"lang": self.lang, "psm": self.psm, "upscale_min_side": self.upscale},
        )
