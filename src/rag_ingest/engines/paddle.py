"""PaddleOCR (Thai) line OCR. Heavy: loads ~1 GB of models on construction."""
from pathlib import Path

from rag_ingest.models import OcrLine, Recognition


class PaddleEngine:
    store_kind = "lines"
    default_output = "ocr"

    def __init__(self, lang: str = "th"):
        from paddleocr import PaddleOCR

        self.lang = lang
        self._ocr = PaddleOCR(use_textline_orientation=True, lang=lang)

    def recognize(self, image: Path) -> Recognition:
        lines = []
        for result in self._ocr.predict(input=str(image)):
            data = result.json
            if callable(data):
                data = data()
            res = data.get("res", data)
            for text, score in zip(res.get("rec_texts", []), res.get("rec_scores", [])):
                lines.append(OcrLine(text, float(score)))
        return Recognition(
            text="\n".join(l.text for l in lines),
            lines=lines,
            model="paddleocr",
            settings={"lang": self.lang},
        )
