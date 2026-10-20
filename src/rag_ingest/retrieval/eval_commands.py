"""`rag-ingest eval-retrieval`: evaluate one index, or the comparison grid (spec 6.3).

Grid: models x --max-tokens x screenshot table mode x context header. The
table dimension is --screenshot-tables, because the tables in the ingested
manuals are all inside screenshots; document tables stay at "keep".

The grid loads one model at a time (never two at once), evaluates every chunk
setting with it, then frees it before the next model. Each grid cell writes
its chunks and index under outputs/index/grid/ (not committed) and shares the
vector cache, so a repeated run only embeds what changed.
"""
import argparse
import gc
import itertools
import time
from dataclasses import asdict

from rag_ingest.chunking import ChunkSettings, CorpusConfig, chunk_corpus, write_run
from rag_ingest.chunking.pipeline import image_fingerprints
from rag_ingest.chunking.tables import KEEP, SCREENSHOT_TABLE_MODES
from rag_ingest.config import DOCUMENT_ROLES, ENCODING, EXCLUDED_DOCUMENTS, Paths
from rag_ingest.embeddings.options import add_embedder_arguments, embedder_options

GRID_MAX_TOKENS = (200, 400, 600)
GRID_MODELS = ("fake", "intfloat/multilingual-e5-small", "intfloat/multilingual-e5-base", "BAAI/bge-m3")
GRID_DIR = "grid"


def short_name(model: str) -> str:
    return model.rsplit("/", 1)[-1].replace("multilingual-", "")


def grid_cells() -> list[ChunkSettings]:
    return [ChunkSettings(max_tokens, 50, KEEP, header, shots)
            for max_tokens, shots, header in itertools.product(GRID_MAX_TOKENS, SCREENSHOT_TABLE_MODES, (True, False))]


def evaluate_cell(paths: Paths, spec, embedder, settings: ChunkSettings, questions, corpus: CorpusConfig):
    from rag_ingest.retrieval.indexing import build_index, load_index
    from rag_ingest.retrieval.report import ConfigResult
    from rag_ingest.retrieval.evaluation import evaluate

    name = f"{short_name(embedder.model)} · {settings.max_tokens} · {settings.screenshot_tables} · " \
           f"{'header' if settings.context_header else 'no header'}"  # no "|": it is a table cell
    slug = f"{short_name(embedder.model)}-{settings.max_tokens}-{settings.screenshot_tables}-{int(settings.context_header)}"
    run = chunk_corpus(corpus, settings, embedder.count_tokens)
    if run.pii_found:
        raise SystemExit(f"Personal data found while chunking for {name}; run `rag-ingest chunk` to see where.")
    chunks_dir = paths.index / GRID_DIR / slug / "chunks"
    write_run(run, chunks_dir, embedder.settings())
    started = time.perf_counter()
    built = build_index(spec, chunks_dir, paths.index, f"{GRID_DIR}/{slug}/index", embedder=embedder)
    index = load_index(built.folder, corpus.ingest_dir)
    results = evaluate(questions, embedder.embed_query, index.search)
    return ConfigResult(name, {"model": embedder.model, **asdict(settings)}, results,
                        time.perf_counter() - started, built.count)


def run_grid(args, paths: Paths, questions) -> list:
    from rag_ingest.retrieval.indexing import EmbedderSpec

    fingerprints, _ = image_fingerprints(EXCLUDED_DOCUMENTS, paths.root / "data")
    corpus = CorpusConfig(paths.resolve(args.ingest), paths.documents, EXCLUDED_DOCUMENTS,
                          frozenset(), DOCUMENT_ROLES, fingerprints)
    configs = []
    for model in args.models:
        spec = EmbedderSpec("fake", {}) if model == "fake" else EmbedderSpec("local", {"model": model})
        embedder = spec.create()
        print(f"Model {model} ...", flush=True)
        configs += [evaluate_cell(paths, spec, embedder, s, questions, corpus) for s in grid_cells()]
        del embedder  # never two models in memory at once
        gc.collect()
    return configs


def run_single(args, paths: Paths, questions) -> list:
    from rag_ingest.retrieval.indexing import latest_index, load_index
    from rag_ingest.retrieval.evaluation import evaluate
    from rag_ingest.retrieval.report import ConfigResult

    folder = paths.resolve(args.index) if args.index else latest_index(paths.index)
    index = load_index(folder, paths.resolve(args.ingest))
    embedder = index.embedder_spec().create()
    results = evaluate(questions, embedder.embed_query, index.search)
    return [ConfigResult(folder.name, index.manifest["chunk_settings"], results, 0.0, index.manifest["count"])]


def run_eval(args: argparse.Namespace, paths: Paths) -> None:
    from rag_ingest.retrieval.evaluation import load_questions
    from rag_ingest.retrieval.report import render, write_csv

    questions = load_questions(paths.resolve(args.questions))
    configs = run_grid(args, paths, questions) if args.grid else run_single(args, paths, questions)
    documents = sorted({doc for q in questions for doc, _ in q.expected})
    # The grid report is the comparison deliverable; a single-index check must not overwrite it.
    default = paths.retrieval_report if args.grid else paths.retrieval_report.with_name("retrieval_report_index.md")
    report = paths.resolve(args.report) if args.report else default
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render(configs, documents), encoding=ENCODING)
    write_csv(report.with_suffix(".csv"), configs)
    print(f"Saved: {report}\nSaved: {report.with_suffix('.csv')}")


def register(add, paths: Paths) -> None:
    p = add("eval-retrieval", run_eval, "Measure whether search finds the right passage.")
    p.add_argument("--questions", default=paths.relative(paths.eval_questions), help="Gold questions (JSONL).")
    p.add_argument("--grid", action="store_true",
                   help="Compare models x --max-tokens {200,400,600} x --screenshot-tables x context header.")
    p.add_argument("--models", nargs="+", default=list(GRID_MODELS), help="Models for --grid ('fake' allowed).")
    p.add_argument("--index", help="Index to evaluate without --grid (default: newest).")
    p.add_argument("--ingest", default=paths.relative(paths.ingest), help="Ingest folder.")
    p.add_argument("--report", help="Report path, .csv written beside it (default: "
                                    "outputs/reports/retrieval_report.md with --grid, else retrieval_report_index.md beside it).")
