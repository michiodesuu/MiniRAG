"""PDF -> Word (.docx), keeping layout and formatting. Thai-aware.

Pipeline per file (one module each):
  preflight   the PDF opens (password), which pages are scans (no text layer,
              they convert to pictures only), which fonts mislabel Thai vowels
  layout      pdf2docx rebuilds each page as a Word section with the same page
              size and margins, fonts, sizes, bold/italic, colours, tables and
              images at their positions
  docx_fixes  what Word needs for Thai that pdf2docx leaves out: complex-script
              font/size/bold/italic on every run, the Thai language tag, and
              the Thai vowels the PDF's fonts mislabel (rag_ingest.thai_glyphs)
  verify      the DOCX text of every page against the PDF; pages, images,
              tables and fonts counted
  report      a markdown summary per file

Fonts are referenced by name, not embedded: the computer opening the .docx
needs the PDF's fonts installed. Pages are converted one after another
(~0.5 GB peak for a 20-page manual); `pages` limits the work further.
Entry points: `rag-ingest convert` and the root-level pdf_to_docx.py.
Needs the `convert` extra (pdf2docx).
"""
from dataclasses import dataclass
from pathlib import Path

from rag_ingest.convert.docx_fixes import apply_thai_fixes
from rag_ingest.convert.layout import rebuild_layout
from rag_ingest.convert.options import parse_pages
from rag_ingest.convert.preflight import inspect_pdf
from rag_ingest.convert.report import render
from rag_ingest.convert.verify import verify


@dataclass(frozen=True)
class ConvertSettings:
    pages: str | None = None      # e.g. "1-3,8"; None = all
    password: str | None = None
    overwrite: bool = False
    repair_vowels: bool = True    # False keeps the PDF's text exactly


def convert_pdf(pdf: Path, docx: Path, settings: ConvertSettings = ConvertSettings()) -> str:
    """Convert one PDF and return its markdown report."""
    if docx.exists() and not settings.overwrite:
        raise FileExistsError(f"{docx} exists; pass --overwrite to replace it")
    facts = inspect_pdf(pdf, settings.password)
    pages = parse_pages(settings.pages, facts.page_count)
    rebuild_layout(pdf, docx, pages, settings.password)
    fixes = apply_thai_fixes(docx, settings.repair_vowels, facts.aa_as_am_fonts)
    selected = [facts.page_texts[i] for i in pages] if pages is not None else facts.page_texts
    return render(pdf, docx, facts, pages, fixes, verify(docx, selected))
