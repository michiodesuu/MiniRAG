"""Turn outputs/ingest/*.md into chunks (spec 3), one document after another.

Per document:
  1. exclusion check (before the file is read)
  2. parse into units; drop table-of-contents pages
  3. sections (Word: "#" headings; PDF: numbered headings from the contents page)
  4. screenshots get their image md5 and OCR source; one already embedded
     elsewhere in the corpus is kept for display but not embedded again
  5. pack each section into chunks (builder.py) within the token budget
  6. guards (guards.py): empty, PII, duplicates
chunk_id = "<document>#p<first page>#c<nn>", numbered in document order before
the guards run, so ids are stable for unchanged input and settings.
"""
from dataclasses import asdict, dataclass, field
from itertools import groupby
from pathlib import Path
import json
from typing import Callable

from rag_ingest.chunking import guards
from rag_ingest.chunking.builder import Draft, SectionPacker, make_blocks
from rag_ingest.chunking.sections import assign_sections, find_table_of_contents, strategy_for
from rag_ingest.chunking.sources import IngestOutput, find_doc_type, nfc_name, sha256_file
from rag_ingest.chunking.tables import COLUMNS, KEEP
from rag_ingest.chunking.text import TableFormat, context_header, nfc
from rag_ingest.chunking.units import Kind, Unit, parse_markdown
from rag_ingest.config import ENCODING

TEXT_KINDS = (Kind.TEXT, Kind.HEADING, Kind.TABLE)
CLI_EXCLUSION = "excluded on the command line (--exclude)"


@dataclass(frozen=True)
class ChunkSettings:
    max_tokens: int = 400
    overlap: int = 50
    tables: str = KEEP
    context_header: bool = True
    screenshot_tables: str = COLUMNS

    @property
    def table_format(self) -> TableFormat:
        return TableFormat(self.tables, self.screenshot_tables)


@dataclass
class DocumentCounts:
    chunks: int = 0
    toc_pages_dropped: list[int] = field(default_factory=list)
    duplicates: int = 0
    duplicate_screenshots: int = 0
    pii_rejections: int = 0
    phone_flags: int = 0
    name_redactions: int = 0
    missing_ocr: int = 0
    empty: int = 0
    oversize: int = 0


@dataclass
class ChunkRun:
    settings: ChunkSettings
    chunks: list[dict] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)  # rule + position, never the value
    flagged: list[dict] = field(default_factory=list)
    dropped: list[dict] = field(default_factory=list)   # duplicates / empty, with pointers
    excluded: list[dict] = field(default_factory=list)
    documents: dict[str, DocumentCounts] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)  # document -> sha256 of its .md

    @property
    def pii_found(self) -> bool:
        return bool(self.rejected)


def image_fingerprints(excluded_files: dict[str, str], search_root: Path) -> tuple[dict[str, str], list[str]]:
    """md5 of every image embedded in each excluded file found under search_root
    (e.g. data/), so a renamed copy is still recognised: (md5 -> file name, names not found)."""
    from rag_ingest.readers import read_pages

    prints, missing = {}, []
    for name in excluded_files:
        found = [p for p in search_root.rglob("*") if p.is_file() and nfc_name(p.name) == nfc_name(name)]
        if not found:
            missing.append(name)
        for path in found:
            prints.update({image.md5: name for page in read_pages(path) for image in page.images})
    return prints, missing


def exclusion_reason(stem: str, images, cli_stems: set[str], excluded_files: dict[str, str],
                     fingerprints: dict[str, str] | None = None) -> str | None:
    """Why a document is excluded, or None. Fails closed:
      - its manifest lists an image of an excluded file (any name: a renamed copy)
      - its stem matches an excluded file, unless the ingest manifest shows only
        other, non-excluded source types for it; a missing manifest, mixed
        evidence or a file found elsewhere (e.g. data/raw) never clears this."""
    copied = sorted({fingerprints[m] for m in images.md5s & set(fingerprints or {})})
    if copied:
        return f"contains images of excluded file {', '.join(copied)}"
    stem = nfc_name(stem)
    if stem in {nfc_name(s) for s in cli_stems}:
        return CLI_EXCLUSION
    source_types = images.source_types
    for name, reason in excluded_files.items():
        path = Path(nfc_name(name))
        excluded_type = path.suffix.lstrip(".")
        proven_other = bool(source_types) and excluded_type not in source_types
        if path.stem == stem and not proven_other:
            return reason
    return None


def mark_pii_units(text: str, units: list[Unit]) -> tuple[list[Unit], list[tuple[int, int]]]:
    """Scan the whole document before chunking, so an ID that a chunk split would cut
    in two is still caught; every unit the ID overlaps is marked."""
    spans = [(start, end) for rule, action, start, end in guards.spans(text) if action == guards.REJECT]
    marked = [u.with_(pii=True) if any(u.start < end and start < u.end for start, end in spans) else u
              for u in units]
    return marked, spans


@dataclass
class Corpus:
    """State shared across documents: what was already embedded."""
    seen_md5: set[str] = field(default_factory=set)
    seen_text: dict[str, str] = field(default_factory=dict)        # embed_text -> chunk_id
    seen_shot_sets: dict[frozenset, str] = field(default_factory=dict)


def link_screenshots(units: list[Unit], images, corpus: Corpus) -> list[Unit]:
    out = []
    for unit in units:
        if unit.kind == Kind.SCREENSHOT:
            md5, source = images.match(unit.page, unit.text)
            repeat = md5 is not None and md5 in corpus.seen_md5
            if md5:
                corpus.seen_md5.add(md5)
            unit = unit.with_(md5=md5, ocr_source=source, duplicate_of_md5=repeat)
        out.append(unit)
    return out


def redact_units(units: list[Unit]) -> tuple[list[Unit], int]:
    """Names out of every unit (and its section) before packing, so no later step
    (split, overlap, header) can hold a name or half of one."""
    out, total = [], 0
    for unit in units:
        text, found = guards.redact_names(unit.text)
        section, in_section = guards.redact_names(unit.section)
        total += found
        out.append(unit.with_(text=text, section=section) if found or in_section else unit)
    return out, total


def chunk_kind(units: list[Unit]) -> str:
    has_text = any(u.kind in TEXT_KINDS for u in units)
    has_ocr = any(u.kind == Kind.SCREENSHOT for u in units)
    return "mixed" if has_text and has_ocr else ("ocr" if has_ocr else "text")


@dataclass
class DocumentContext:
    stem: str
    doc_type: str
    role: str | None
    sha256: str


def make_chunk(draft: Draft, header: str, index: int, doc: DocumentContext,
               count: Callable[[str], int]) -> dict:
    units = draft.units
    body = nfc(draft.body)
    embed_text = guards.redact_names(f"{header}\n{body}" if header and body else body)[0]  # backstop
    shots = [u for u in units if u.kind == Kind.SCREENSHOT]
    pages = [u.page for u in units]
    return {
        "chunk_id": f"{doc.stem}#p{min(pages)}#c{index:02d}",
        "document": doc.stem, "doc_type": doc.doc_type, "role": doc.role,
        "section": units[0].section,
        "page_start": min(pages), "page_end": max(pages),
        "kind": chunk_kind(units),
        "ocr_sources": sorted({u.ocr_source for u in shots if u.ocr_source}),
        "screenshot_md5": [u.md5 for u in shots if u.md5],
        "missing_ocr": sum(1 for u in units if u.kind == Kind.PLACEHOLDER),
        "duplicate_screenshots": sum(1 for u in shots if u.duplicate_of_md5),
        "tokens": count(embed_text),
        "embed_text": embed_text,
        "display_text": guards.redact_names(draft.display)[0],
        "source_sha256": doc.sha256,
        "char_start": min(u.start for u in units), "char_end": max(u.end for u in units),
    }


class Chunker:
    def __init__(self, settings: ChunkSettings, count: Callable[[str], int]):
        self.settings, self.count = settings, count

    def header(self, stem: str, section: str) -> str:
        return guards.redact_names(context_header(stem, section))[0] if self.settings.context_header else ""

    def room_for(self, header: str) -> Callable[[str], int]:
        """Tokens left in the budget for a chunk body under this header."""
        def room(body: str) -> int:
            text = f"{header}\n{body}" if header else body
            return self.settings.max_tokens - self.count(text)
        return room

    def document_drafts(self, units: list[Unit], stem: str,
                        counts: DocumentCounts) -> list[tuple[Draft, str]]:
        drafts = []
        for section, group in groupby(units, key=lambda u: u.section):
            header = self.header(stem, section)
            packer = SectionPacker(self.room_for(header), self.count, self.settings.overlap,
                                   self.settings.table_format)
            drafts += [(d, header) for d in packer.pack(make_blocks(list(group), self.settings.table_format))]
            counts.oversize += packer.oversize
        return drafts


def apply_guards(chunk: dict, run: ChunkRun, corpus: Corpus, counts: DocumentCounts,
                 has_pii_unit: bool = False) -> bool:
    """True if the chunk is kept."""
    findings = guards.scan({"embed_text": chunk["embed_text"], "display_text": chunk["display_text"]})
    if has_pii_unit and not any(f.action == guards.REJECT for f in findings):
        # caught by the whole-document scan only (e.g. the ID spans a chunk split)
        findings.append(guards.Finding(guards.THAI_ID, guards.REJECT, "source_markdown",
                                       chunk["char_start"], chunk["char_end"] - chunk["char_start"]))
    base = {"chunk_id": chunk["chunk_id"], "document": chunk["document"]}
    run.flagged += [{**base, **asdict(f)} for f in findings if f.action == guards.FLAG]
    counts.phone_flags += sum(1 for f in findings if f.action == guards.FLAG)
    rejected = [f for f in findings if f.action == guards.REJECT]
    if rejected:
        run.rejected += [{**base, **asdict(f)} for f in rejected]
        counts.pii_rejections += 1
        return False
    reason, pointer = drop_reason(chunk, corpus)
    if reason:
        run.dropped.append({**base, "rule": reason, "duplicate_of": pointer})
        if reason == guards.EMPTY:
            counts.empty += 1
        else:
            counts.duplicates += 1
        return False
    corpus.seen_text[chunk["embed_text"]] = chunk["chunk_id"]
    if chunk["kind"] == "ocr":
        corpus.seen_shot_sets[frozenset(chunk["screenshot_md5"])] = chunk["chunk_id"]
    return True


def drop_reason(chunk: dict, corpus: Corpus) -> tuple[str | None, str | None]:
    if not chunk["embed_text"].strip():  # make_chunk adds no header to an empty body
        return guards.EMPTY, None
    if chunk["embed_text"] in corpus.seen_text:
        return guards.DUPLICATE_TEXT, corpus.seen_text[chunk["embed_text"]]
    shots = frozenset(chunk["screenshot_md5"])
    if chunk["kind"] == "ocr" and shots in corpus.seen_shot_sets:
        return guards.DUPLICATE_SCREENSHOTS, corpus.seen_shot_sets[shots]
    return None, None


@dataclass(frozen=True)
class CorpusConfig:
    ingest_dir: Path
    documents_dir: Path
    excluded_files: dict[str, str]
    cli_excluded: frozenset[str]
    roles: dict[str, str]
    fingerprints: dict[str, str] = field(default_factory=dict)  # image md5 -> excluded file


def chunk_corpus(config: CorpusConfig, settings: ChunkSettings,
                 count: Callable[[str], int]) -> ChunkRun:
    output, run, corpus = IngestOutput.load(config.ingest_dir), ChunkRun(settings), Corpus()
    chunker = Chunker(settings, count)
    for path in output.markdown_files():
        images = output.images(path.stem)
        doc_type = find_doc_type(path.stem, images, config.documents_dir)
        reason = exclusion_reason(path.stem, images, set(config.cli_excluded),
                                  config.excluded_files, config.fingerprints)
        if reason:
            run.excluded.append({"document": path.stem, "doc_type": doc_type, "reason": reason})
            continue
        doc = DocumentContext(path.stem, doc_type, config.roles.get(path.stem), sha256_file(path))
        chunk_document(path, images, doc, chunker, run, corpus)
    return run


def chunk_document(path: Path, images, doc: DocumentContext, chunker: Chunker,
                   run: ChunkRun, corpus: Corpus) -> None:
    counts = run.documents.setdefault(doc.stem, DocumentCounts())
    run.sources[doc.stem] = doc.sha256
    text = path.read_text(encoding=ENCODING)
    units, _ = mark_pii_units(text, parse_markdown(text).units)
    toc = find_table_of_contents(units)
    counts.toc_pages_dropped = sorted(toc.pages)
    units = assign_sections([u for u in units if u.page not in toc.pages], strategy_for(doc.doc_type, toc))
    units = link_screenshots(units, images, corpus)  # needs the OCR text as ingested
    units, counts.name_redactions = redact_units(units)
    for index, (draft, header) in enumerate(chunker.document_drafts(units, doc.stem, counts)):
        chunk = make_chunk(draft, header, index, doc, chunker.count)
        counts.missing_ocr += chunk["missing_ocr"]
        counts.duplicate_screenshots += chunk["duplicate_screenshots"]
        if apply_guards(chunk, run, corpus, counts, any(u.pii for u in draft.units)):
            run.chunks.append(chunk)
            counts.chunks += 1


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding=ENCODING)


def write_run(run: ChunkRun, out_dir: Path, tokenizer: dict, extra: dict | None = None) -> Path:
    """chunks.jsonl, rejected.jsonl (rule + position only) and manifest.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = out_dir / "chunks.jsonl"
    write_jsonl(chunks_path, run.chunks)
    write_jsonl(out_dir / "rejected.jsonl", run.rejected)
    manifest = {
        "settings": {**asdict(run.settings), "tokenizer": tokenizer},
        "chunks": len(run.chunks),
        "chunks_sha256": sha256_file(chunks_path),
        "documents": {doc: asdict(counts) for doc, counts in run.documents.items()},
        "excluded": run.excluded,
        "dropped": run.dropped,
        "flagged": run.flagged,
        "sources": run.sources,
        **(extra or {}),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                           encoding=ENCODING)
    return chunks_path


def summary(run: ChunkRun) -> str:
    """Console summary: counts only, never chunk text or matched values."""
    lines = [f"Chunks: {len(run.chunks)} from {len(run.documents)} document(s)"]
    for doc, c in run.documents.items():
        lines.append(f"  {doc}: {c.chunks} chunks, contents pages dropped {c.toc_pages_dropped or '-'}, "
                     f"duplicates {c.duplicates}, repeated screenshots {c.duplicate_screenshots}, "
                     f"missing OCR {c.missing_ocr}, PII rejected {c.pii_rejections}, "
                     f"phone flags {c.phone_flags}, names redacted {c.name_redactions}, oversize {c.oversize}")
    lines += [f"Excluded: {e['document']} ({e['reason']})" for e in run.excluded]
    lines += [f"Rejected: {r['chunk_id']} [{r['rule']}] {r['field']} @{r['start']}+{r['length']}"
              for r in run.rejected]
    return "\n".join(lines)
