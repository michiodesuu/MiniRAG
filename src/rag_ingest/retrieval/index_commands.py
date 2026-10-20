"""`rag-ingest embed` and `rag-ingest search`: argument wiring for the index stage."""
import argparse
import sys
import time

from rag_ingest.config import Paths
from rag_ingest.embeddings.options import add_embedder_arguments, embedder_options

SNIPPET_CHARS = 160


def run_embed(args: argparse.Namespace, paths: Paths) -> None:
    from rag_ingest.retrieval.indexing import EmbedderSpec, build_index

    spec = EmbedderSpec(args.embedder, embedder_options(args))
    started = time.perf_counter()
    built = build_index(spec, paths.resolve(args.chunks), paths.resolve(args.index_root), args.run,
                        use_cache=not args.no_cache)
    print(f"Indexed {built.count} chunks ({built.computed} embedded now, "
          f"{built.count - built.computed} from cache) in {time.perf_counter() - started:.1f} s")
    print(f"Saved: {built.folder}")


def snippet(text: str) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= SNIPPET_CHARS else flat[:SNIPPET_CHARS] + "…"


def run_search(args: argparse.Namespace, paths: Paths) -> None:
    from rag_ingest.retrieval.indexing import StaleIndexError, latest_index, load_index

    folder = paths.resolve(args.index) if args.index else latest_index(paths.index)
    try:
        index = load_index(folder, paths.resolve(args.ingest))
    except StaleIndexError as error:
        sys.exit(str(error))
    embedder = index.embedder_spec().create()
    filters = {"role": args.role, "document": args.document, "doc_type": args.doc_type}
    hits = index.search(embedder.embed_query(args.query), args.k, filters)
    print(f"Index: {folder.name} ({index.manifest['embedder']['settings']['model']})")
    for rank, hit in enumerate(hits, 1):
        print(f"{rank}. {hit.score:.3f}  {hit.chunk_id}  p.{hit.pages}  {hit.meta.get('section') or '-'}")
        print(f"   {snippet(index.texts.get(hit.chunk_id, ''))}")
    if not hits:
        print("No results (check the filters).")


def register(add, paths: Paths) -> None:
    p = add("embed", run_embed, "Embed outputs/chunks/chunks.jsonl into outputs/index/<run>/.")
    p.add_argument("--chunks", default=paths.relative(paths.chunks), help="Chunk folder.")
    p.add_argument("--index-root", default=paths.relative(paths.index), help="Where index runs go.")
    p.add_argument("--run", help="Index folder name (default: <model>__<settings hash>).")
    p.add_argument("--no-cache", action="store_true", help="Embed every chunk again.")
    add_embedder_arguments(p)

    p = add("search", run_search, "Search the index (manual check of retrieval).")
    p.add_argument("query", help="The question, e.g. \"ถ้ารายการถูกอนุมัติแล้วจะแก้ไขได้อย่างไร\".")
    p.add_argument("-k", type=int, default=5, help="Number of results.")
    p.add_argument("--role", help="Only this role, e.g. branch_manager.")
    p.add_argument("--document", help="Only this document (stem).")
    p.add_argument("--doc-type", choices=["pdf", "docx"], help="Only this source type.")
    p.add_argument("--index", help="Index folder (default: the newest under outputs/index).")
    p.add_argument("--ingest", default=paths.relative(paths.ingest),
                   help="Ingest folder, for the staleness check.")
