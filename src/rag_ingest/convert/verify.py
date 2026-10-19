"""Check the DOCX against the PDF: text per page, and what it contains."""
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from rag_ingest.readers.word import ooxml as x
from rag_ingest.textproc import normalize_for_scoring

LOW_SIMILARITY = 0.95  # pages below this text similarity are flagged in the report


@dataclass
class Verification:
    sections: int
    images: int
    tables: int
    fonts: list[str]
    page_similarity: list[float]  # one per converted page

    @property
    def low_pages(self) -> list[int]:
        return [i for i, s in enumerate(self.page_similarity, 1) if s < LOW_SIMILARITY]

    @property
    def mean_similarity(self) -> float:
        scores = self.page_similarity
        return sum(scores) / len(scores) if scores else 0.0


def docx_page_texts(body) -> list[str]:
    """Text per page: pdf2docx ends every page's section with a paragraph-level w:sectPr."""
    pages, current = [], []
    for paragraph in body.iter(x.P):
        current.append("".join(t.text or "" for t in paragraph.iter(x.T)))
        ppr = paragraph.find(x.PPR)
        if ppr is not None and ppr.find(x.SECTPR) is not None:
            pages.append("\n".join(current))
            current = []
    if current or not pages:
        pages.append("\n".join(current))
    return pages


def similarity(a: str, b: str) -> float:
    a, b = normalize_for_scoring(a), normalize_for_scoring(b)
    if not a and not b:
        return 1.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def verify(docx: Path, pdf_texts: list[str]) -> Verification:
    from docx import Document

    body = Document(str(docx)).element.body
    pages = docx_page_texts(body)
    fonts = sorted({a for f in body.iter(x.RFONTS) if (a := f.get(x.q("w:ascii")))})
    return Verification(sections=sum(1 for _ in body.iter(x.SECTPR)),
                        images=sum(1 for _ in body.iter(x.BLIP)),
                        tables=sum(1 for _ in body.iter(x.TBL)),
                        fonts=fonts,
                        page_similarity=[similarity(p, d) for p, d in zip(pdf_texts, pages)])
