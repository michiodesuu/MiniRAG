"""`rag-ingest chunk`: argument wiring for the chunking stage."""
import argparse
import sys

from rag_ingest.chunking import guards
from rag_ingest.chunking.pipeline import (ChunkSettings, CorpusConfig, chunk_corpus,
                                          image_fingerprints, summary, write_run)
from rag_ingest.chunking.tables import COLUMNS, SCREENSHOT_TABLE_MODES, TABLE_MODES
from rag_ingest.config import DOCUMENT_ROLES, EXCLUDED_DOCUMENTS, Paths
from rag_ingest.embeddings.options import add_embedder_arguments, embedder_options

DEFAULT_DENYLIST = "data/name_denylist.txt"
PII_STOP = ("Stopped: personal data found (positions above, values never written). "
            "Those chunks are not in chunks.jsonl; pass --allow-pii to finish with exit code 0 anyway.")


def install_denylist(args: argparse.Namespace, paths: Paths) -> int:
    path = paths.resolve(args.name_denylist or DEFAULT_DENYLIST)
    if not path.is_file():
        if args.name_denylist:
            sys.exit(f"Name deny-list not found: {path}")
        print(f"No name deny-list ({DEFAULT_DENYLIST}): only titles and labels are redacted", file=sys.stderr)
        return 0
    return guards.load_name_denylist(path)


def run_chunk(args: argparse.Namespace, paths: Paths) -> None:
    from rag_ingest.embeddings import create_embedder

    denylisted = install_denylist(args, paths)

    embedder = create_embedder(args.embedder, **embedder_options(args))  # its tokenizer counts
    fingerprints, not_found = image_fingerprints(EXCLUDED_DOCUMENTS, paths.root / "data")
    config = CorpusConfig(paths.resolve(args.input), paths.documents, EXCLUDED_DOCUMENTS,
                          frozenset(args.exclude), DOCUMENT_ROLES, fingerprints)
    settings = ChunkSettings(args.max_tokens, args.overlap, args.tables, not args.no_context_header,
                             args.screenshot_tables)
    run = chunk_corpus(config, settings, embedder.count_tokens)
    out = write_run(run, paths.resolve(args.output), embedder.settings(),
                    {"exclusion_fingerprints": {"images": len(fingerprints), "files_not_found": not_found},
                     "name_denylist": {"entries": denylisted}})
    print(summary(run))
    if not_found:
        print(f"Warning: excluded file(s) not found, only excluded by name: {', '.join(not_found)}",
              file=sys.stderr)
    print(f"Saved: {out}")
    if run.pii_found and not args.allow_pii:
        sys.exit(PII_STOP)


def register(add, paths: Paths) -> None:
    p = add("chunk", run_chunk, "Split outputs/ingest/*.md into chunks for embedding.")
    p.add_argument("--input", default=paths.relative(paths.ingest), help="Ingest output folder.")
    p.add_argument("--output", default=paths.relative(paths.chunks), help="Chunk output folder.")
    p.add_argument("--max-tokens", type=int, default=400,
                   help="Token budget per chunk, header included (model tokenizer).")
    p.add_argument("--overlap", type=int, default=50, help="Tokens carried from the previous chunk.")
    p.add_argument("--tables", choices=TABLE_MODES, default="keep",
                   help="Tables in the document's own text. keep: Markdown tables; "
                        "flatten: one 'header: value' line per row.")
    p.add_argument("--screenshot-tables", choices=SCREENSHOT_TABLE_MODES, default=COLUMNS,
                   help="Tables in screenshot OCR: keep, flatten, or columns (header row only; "
                        "drops the sample records a list screen shows).")
    p.add_argument("--no-context-header", action="store_true",
                   help="Do not prefix chunks with '<document> › <section>'.")
    p.add_argument("--exclude", nargs="*", default=[], metavar="STEM",
                   help="More documents to leave out (reported, never silently skipped).")
    p.add_argument("--name-denylist", metavar="FILE",
                   help=f"Names to redact wherever they appear, one per line (default {DEFAULT_DENYLIST} "
                        "if it exists; it is git-ignored because it is personal data).")
    p.add_argument("--allow-pii", action="store_true",
                   help="Exit 0 even if personal data was found (those chunks stay out).")
    add_embedder_arguments(p)
