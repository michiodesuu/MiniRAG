"""Domain objects shared by every stage."""
from dataclasses import dataclass, field
from pathlib import Path
import re

_NAME = re.compile(r"(?P<document>.+)_page_(?P<page>\d+)_image_(?P<index>\d+)$")


@dataclass(frozen=True)
class ImageRef:
    """Where an embedded image came from. Its `stem` names every result file."""
    document: str  # PDF file stem
    page: int      # 1-based
    index: int     # 1-based, in page.get_images() order

    @property
    def stem(self) -> str:
        return f"{self.document}_page_{self.page}_image_{self.index}"

    @classmethod
    def parse(cls, stem: str) -> "ImageRef | None":
        match = _NAME.match(Path(stem).stem)
        if not match:
            return None
        return cls(match["document"], int(match["page"]), int(match["index"]))


@dataclass(frozen=True)
class OcrLine:
    text: str
    confidence: float


@dataclass
class Recognition:
    """What an OCR engine returns for one image."""
    text: str
    lines: list[OcrLine] | None = None  # line engines (Tesseract, PaddleOCR) only
    model: str = ""
    settings: dict = field(default_factory=dict)
