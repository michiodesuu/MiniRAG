"""Domain objects shared by every stage."""
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import re

_NAME = re.compile(r"(?P<document>.+)_page_(?P<page>\d+)_image_(?P<index>\d+)$")


@dataclass(frozen=True)
class ImageRef:
    """Where an embedded image came from. Its `stem` names every result file."""
    document: str  # source file stem (PDF or Word)
    page: int      # 1-based; approximate for Word, which has no fixed pages
    index: int     # 1-based, in reading order on the page

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
    lines: list[OcrLine] | None = None  # line engines (Tesseract, EasyOCR) only
    model: str = ""
    settings: dict = field(default_factory=dict)
    prompt_file: Path | None = None  # vision engines: the prompt actually sent


@dataclass(frozen=True)
class TextBlock:
    y: float  # position on the page (PDF points), or reading order (Word)
    x: float
    text: str


@dataclass(frozen=True)
class EmbeddedImage:
    ref: ImageRef
    data: bytes
    ext: str
    width: int   # pixels; 0 if the format cannot be decoded (e.g. EMF/WMF)
    height: int
    y: float | None  # same axis as TextBlock.y; None if the image is not drawn
    x: float | None

    @property
    def md5(self) -> str:
        return hashlib.md5(self.data).hexdigest()

    @property
    def short_side(self) -> int:
        return min(self.width, self.height)

    @property
    def decodable(self) -> bool:
        return self.width > 0 and self.height > 0


@dataclass
class Page:
    """One page of any source document, in the shape every later stage expects."""
    number: int  # 1-based
    height: float
    raw_text: str
    blocks: list[TextBlock]
    images: list[EmbeddedImage]
