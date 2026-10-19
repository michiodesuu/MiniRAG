"""Word tables: read the grid (merged cells kept aligned), render it as markdown.

Reading and rendering are separate so another output form (HTML, CSV) only
needs a new renderer.
"""
from typing import Callable, Iterator

from rag_ingest.readers.word import ooxml as x

_CONTAINERS = (x.SDT, x.SDT_CONTENT)


def cells(table) -> Iterator:
    for row in table.iterchildren(x.TR):
        yield from row.iterchildren(x.TC)


def cell_paragraphs(cell) -> Iterator:
    """Paragraphs of a cell in order, including nested tables, excluding text boxes."""
    for child in cell:
        if child.tag == x.P:
            yield child
        elif child.tag == x.TBL:
            for nested in cells(child):
                yield from cell_paragraphs(nested)
        elif child.tag in _CONTAINERS:
            yield from cell_paragraphs(child)


def cell_layout(cell) -> tuple[int, bool]:
    """(columns spanned, continues a vertical merge from the row above)."""
    props = cell.find(x.TCPR)
    span = int(x.val(props, "w:gridSpan") or 1)
    merge = None if props is None else props.find(x.q("w:vMerge"))
    return span, merge is not None and merge.get(x.VAL) != "restart"


def read_grid(table, cell_text: Callable[[object], str]) -> list[list[str]]:
    """Rows of cell texts. A merged cell's text sits in its first column / top row."""
    grid = []
    for row in table.iterchildren(x.TR):
        texts = []
        for cell in row.iterchildren(x.TC):
            span, continued = cell_layout(cell)
            texts.append("" if continued else cell_text(cell))
            texts.extend([""] * (span - 1))
        grid.append(texts)
    return grid


def escape_cell(text: str) -> str:
    """One markdown cell: no pipes or line breaks that would break the row."""
    return text.replace("|", "\\|").replace("\n", "<br>").replace("\t", " ")


def markdown_table(rows: list[list[str]]) -> str:
    rows = [r for r in rows if any(c.strip() for c in r)]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)
