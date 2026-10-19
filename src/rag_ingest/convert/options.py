"""Which files and pages to convert, and where the results go."""
from pathlib import Path


def parse_pages(spec: str | None, page_count: int) -> list[int] | None:
    """"1-3,8" -> [0, 1, 2, 7] (0-based, sorted, unique). None means every page."""
    if not spec:
        return None
    pages: set[int] = set()
    for part in spec.split(","):
        first, _, last = part.strip().partition("-")
        start, end = int(first), int(last or first)
        if not 1 <= start <= end <= page_count:
            raise ValueError(f"page range {part!r} is outside 1-{page_count}")
        pages.update(range(start - 1, end))
    return sorted(pages)


def output_path(pdf: Path, output: Path | None, output_dir: Path) -> Path:
    return output if output else output_dir / f"{pdf.stem}.docx"


def collect_pdfs(inputs: list[Path]) -> list[Path]:
    pdfs: list[Path] = []
    for item in inputs:
        if item.is_dir():
            pdfs.extend(sorted(item.glob("*.pdf")))
        elif item.suffix.lower() == ".pdf" and item.is_file():
            pdfs.append(item)
        else:
            raise FileNotFoundError(f"Not a PDF file or folder: {item}")
    if not pdfs:
        raise FileNotFoundError("No PDF files found in the given inputs")
    return pdfs
