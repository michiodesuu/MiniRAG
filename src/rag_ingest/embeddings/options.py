"""Embedder command-line options, shared by chunk / embed / search / eval-retrieval,
and the options each embedder takes (like cli.engine_options for OCR engines)."""
import argparse

from rag_ingest.embeddings import EMBEDDERS
from rag_ingest.embeddings.profiles import DEFAULT_BATCH_SIZE, DEFAULT_MODEL, PROFILES


def add_embedder_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--embedder", choices=list(EMBEDDERS), default="local",
                        help="local (sentence-transformers, CPU), api (internal OpenAI-compatible "
                             "endpoint) or fake (hashing; tests/baseline).")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Model name (default {DEFAULT_MODEL}). Profiled: {', '.join(PROFILES)}.")
    parser.add_argument("--revision", help="Model commit hash (default: the pinned one in profiles.py).")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help="Passages per forward pass (local).")
    parser.add_argument("--max-seq-length", type=int,
                        help="Tokens kept per input; longer inputs are truncated (default per model).")
    parser.add_argument("--api-url", help="api: full /v1/embeddings URL (or env EMBED_API_URL).")
    parser.add_argument("--api-model", help="api: model name sent to the server (or env EMBED_API_MODEL).")
    parser.add_argument("--tokenizer", help="api: Hugging Face tokenizer matching the served model, "
                                            "for exact token counts (default: approximate).")


def embedder_options(args: argparse.Namespace) -> dict:
    if args.embedder == "fake":
        return {}
    if args.embedder == "local":
        return {"model": args.model, "revision": args.revision, "batch_size": args.batch_size,
                "max_seq_length": args.max_seq_length}
    from rag_ingest.embeddings.api import api_options
    return api_options(args)
