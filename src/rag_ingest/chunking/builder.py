"""Pack one section's units into chunks within a token budget (Builder).

Rules (spec 3.2):
  - Blocks: a unit plus the screenshots/placeholders right after it, so a
    screenshot stays with the instruction that explains it.
  - Greedy packing: add the next block while the chunk's embed text, with its
    context header, stays within --max-tokens (counted by the embedder's own
    tokenizer, via `fits`).
  - A block that does not fit on its own is split into its units; a unit that
    does not fit is split by lines or table rows, repeating the table header.
    Those parts leave room for the overlap, so long passages still overlap.
    A single line longer than the budget is kept whole (never cut mid-word)
    and reported as oversize.
  - Overlap: a new chunk starts with the last lines of the previous chunk's
    final text block, up to --overlap tokens; whole lines, or whole
    space-separated phrases of one long line, never half a word. Screenshots,
    tables and units holding a rejected personal number are never used as overlap.
  - One packer per section: nothing crosses a section or document boundary.
"""
from dataclasses import dataclass, field
from typing import Callable

from rag_ingest.chunking.text import Segment, TableFormat, display_form, unit_segments
from rag_ingest.chunking.units import Kind, Unit

ATTACHED_KINDS = (Kind.SCREENSHOT, Kind.PLACEHOLDER)
OVERLAP_KINDS = (Kind.TEXT, Kind.HEADING)


@dataclass(frozen=True)
class Block:
    units: tuple[Unit, ...]
    segments: tuple[Segment, ...]
    display: str
    is_part: bool = False  # a piece of a unit that had to be split

    @property
    def embed(self) -> str:
        return "\n".join(s.text for s in self.segments)


def unit_block(unit: Unit, fmt: TableFormat) -> Block:
    segments = unit_segments(unit, fmt)
    embedded = () if unit.duplicate_of_md5 else tuple(segments)  # shown, not embedded again
    return Block((unit,), embedded, display_form(unit, segments))


def make_blocks(units: list[Unit], fmt: TableFormat) -> list[Block]:
    groups: list[list[Unit]] = []
    for unit in units:
        if unit.kind in ATTACHED_KINDS and groups:
            groups[-1].append(unit)
        else:
            groups.append([unit])
    return [merge_blocks([unit_block(u, fmt) for u in group]) for group in groups]


def merge_blocks(blocks: list[Block]) -> Block:
    units = tuple(u for b in blocks for u in b.units)
    segments = tuple(s for b in blocks for s in b.segments)
    return Block(units, segments, "\n\n".join(b.display for b in blocks))


@dataclass
class Draft:
    blocks: list[Block]
    overlap: str = ""

    @property
    def units(self) -> list[Unit]:
        seen, out = set(), []
        for unit in (u for b in self.blocks for u in b.units):
            if id(unit) not in seen:
                seen.add(id(unit))
                out.append(unit)
        return out

    @property
    def body(self) -> str:
        return join_body(self.blocks, self.overlap)

    @property
    def display(self) -> str:
        return "\n\n".join(b.display for b in self.blocks)


def join_body(blocks: list[Block], overlap: str = "") -> str:
    return "\n".join(([overlap] if overlap else []) + [b.embed for b in blocks if b.embed])


@dataclass
class SectionPacker:
    room: Callable[[str], int]        # tokens left in the budget for this body (< 0: over)
    count: Callable[[str], int]       # tokens of a text
    overlap_tokens: int
    table_format: TableFormat
    drafts: list[Draft] = field(default_factory=list)
    oversize: int = 0
    _current: list[Block] = field(default_factory=list)
    _carry: str = ""

    def fits(self, body: str) -> bool:
        return self.room(body) >= 0

    def pack(self, blocks: list[Block]) -> list[Draft]:
        for block in blocks:
            for piece in self.pieces(block):
                self._add(piece)
        if self._current:
            self._close()
        return self.drafts

    def pieces(self, block: Block) -> list[Block]:
        """The block, or smaller blocks that each fit the budget on their own."""
        if self.fits(block.embed):
            return [block]
        if len(block.units) > 1:
            return [p for u in block.units for p in self.pieces(unit_block(u, self.table_format))]
        return self._split_lines(block)

    def _split_lines(self, block: Block) -> list[Block]:
        reserve = self.overlap_tokens + 1 if self.overlap_tokens else 0  # +1: the joining newline
        parts: list[list[str]] = []
        lines: list[str] = []
        for segment in block.segments:
            start = list(segment.header) if not lines else lines
            trial = start + [segment.text]
            if lines and self.room("\n".join(trial)) < reserve:
                parts.append(lines)
                trial = list(segment.header) + [segment.text]  # repeat the table header
            lines = trial
        if lines:
            parts.append(lines)
        self.oversize += sum(1 for p in parts if not self.fits("\n".join(p)))
        return [part_block(block, p) for p in parts]

    def _add(self, piece: Block) -> None:
        if self._current and self.fits(join_body(self._current + [piece], self._carry)):
            self._current.append(piece)
            return
        if self._current:
            self._close()
        self._current = [piece]
        if not self.fits(join_body(self._current, self._carry)):
            self._carry = ""  # the overlap would push the piece over the budget

    def _close(self) -> None:
        self.drafts.append(Draft(self._current, self._carry))
        self._carry = self._overlap_from(self._current[-1])
        self._current = []

    def _overlap_from(self, block: Block) -> str:
        if not self.overlap_tokens or not all(u.kind in OVERLAP_KINDS and not u.pii
                                               for u in block.units):
            return ""  # never screenshots/tables, never text holding personal data
        lines = [s.text for s in block.segments]
        taken = trailing(lines, "\n", self.count, self.overlap_tokens)
        if not taken and lines:  # one long line: whole space-separated phrases instead
            taken = trailing(lines[-1].split(" "), " ", self.count, self.overlap_tokens)
        return taken


def trailing(pieces: list[str], joiner: str, count: Callable[[str], int], budget: int) -> str:
    """The longest suffix of whole pieces whose joined text fits the budget."""
    chosen: list[str] = []
    for piece in reversed(pieces):
        if count(joiner.join([piece] + chosen)) > budget:
            break
        chosen.insert(0, piece)
    return joiner.join(chosen).strip()


def part_block(block: Block, lines: list[str]) -> Block:
    text = "\n".join(lines)
    unit = block.units[0]
    display = display_form(unit.with_(text=text)) if unit.kind == Kind.SCREENSHOT else text
    return Block(block.units, tuple(Segment(line) for line in lines), display, is_part=True)
