"""Builder for documents that are a flow of text and images (e.g. Word), not fixed pages.

A reader walks its format and calls add_text / add_image / new_page; the
builder numbers pages and images and assigns reading-order positions, so
readers never construct Page objects themselves.
"""
from io import BytesIO

from PIL import Image

from rag_ingest.models import EmbeddedImage, ImageRef, Page, TextBlock


def pixel_size(data: bytes) -> tuple[int, int]:
    """(width, height), or (0, 0) for formats Pillow cannot decode here (EMF, WMF)."""
    try:
        with Image.open(BytesIO(data)) as img:
            return img.size
    except Exception:
        return 0, 0


class PageBuilder:
    def __init__(self, document_stem: str):
        self.stem = document_stem
        self.order = 0
        self.pages: list[Page] = []
        self.new_page()

    @property
    def page(self) -> Page:
        return self.pages[-1]

    @property
    def page_has_content(self) -> bool:
        return bool(self.page.blocks or self.page.images)

    def new_page(self) -> None:
        self.pages.append(Page(len(self.pages) + 1, 0.0, "", [], []))

    def _next_position(self) -> float:
        self.order += 1
        return float(self.order)

    def add_text(self, text: str) -> None:
        if text.strip():
            self.page.blocks.append(TextBlock(self._next_position(), 0.0, text.strip("\n")))

    def add_image(self, data: bytes, ext: str) -> None:
        width, height = pixel_size(data)
        ref = ImageRef(self.stem, self.page.number, len(self.page.images) + 1)
        self.page.images.append(EmbeddedImage(ref, data, ext, width, height,
                                            self._next_position(), 0.0))

    def finish(self) -> list[Page]:
        for page in self.pages:
            page.raw_text = "\n".join(b.text for b in page.blocks)
            page.height = float(self.order + 1)  # below every position: "end of page"
        return self.pages
