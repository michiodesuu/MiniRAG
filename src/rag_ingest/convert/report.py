"""The markdown summary printed (and optionally saved) for each converted file."""
from pathlib import Path

from rag_ingest.convert.docx_fixes import FixCounts
from rag_ingest.convert.preflight import PdfFacts
from rag_ingest.convert.verify import LOW_SIMILARITY, Verification


def _defective_note(facts: PdfFacts) -> str:
    names = sorted(k.family + (" Bold" if k.bold else "") for k in facts.aa_as_am_fonts)
    return f" (fonts labelling 'า' as 'ำ': {', '.join(names)})" if names else ""


def render(pdf: Path, docx: Path, facts: PdfFacts, pages: list[int] | None,
           fixes: FixCounts, check: Verification) -> str:
    converted = len(pages) if pages is not None else facts.page_count
    lines = [
        f"# {pdf.name} -> {docx.name}",
        "",
        f"- Pages: {converted} of {facts.page_count} converted; DOCX sections: {check.sections}",
        f"- Images: PDF placements {facts.images}, DOCX pictures {check.images}"
        " (overlapping images, e.g. blur patches on a screenshot, become one picture)",
        f"- Tables in DOCX: {check.tables} (pdf2docx also uses borderless tables to hold layout)",
        f"- Fonts: {', '.join(check.fonts) or '-'} (must be installed where the DOCX is opened)",
        f"- Thai fixes: complex-script font/size/bold set on {fixes.complex_script_runs} of "
        f"{fixes.runs} runs; Thai language on {fixes.thai_runs}",
        f"- Thai vowels the PDF mislabels: 'ำ' restored {fixes.am_restored}x, "
        f"'า' restored {fixes.aa_restored}x{_defective_note(facts)}",
        f"- Text match with the PDF: mean {check.mean_similarity:.1%}",
    ]
    if facts.scanned_pages:
        lines.append(f"- Scanned pages (no text layer, kept as pictures): {facts.scanned_pages}")
    if check.low_pages:
        lines.append(f"- Pages to check by eye (text match < {LOW_SIMILARITY:.0%}): "
                     + ", ".join(f"{p} ({check.page_similarity[p - 1]:.0%})" for p in check.low_pages))
    return "\n".join(lines)
