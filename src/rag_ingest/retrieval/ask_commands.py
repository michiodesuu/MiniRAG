"""The end-to-end `rag-ingest ask` command."""
import argparse
import sys
from uuid import uuid4

from rag_ingest.config import Paths


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def run_ask(args: argparse.Namespace, paths: Paths) -> None:
    from rag_ingest.databases.postgres import PostgresChunkStore
    from rag_ingest.databases.settings import read_env_file
    from rag_ingest.engines.chat import ChatClient
    from rag_ingest.retrieval.access import filter_authorized_hits
    from rag_ingest.retrieval.answer import generate_grounded_answer
    from rag_ingest.retrieval.conversation import rewrite_query
    from rag_ingest.retrieval.indexing import StaleIndexError, latest_index, load_index
    from rag_ingest.retrieval.reranker import RerankerClient

    read_env_file(paths.root / ".env")
    folder = paths.resolve(args.index) if args.index else latest_index(paths.index)
    try:
        index = load_index(folder, paths.resolve(args.ingest))
    except StaleIndexError as error:
        sys.exit(str(error))

    if not args.query.strip():
        raise ValueError("Query must not be empty")
    chat = ChatClient.from_env()
    reranker = RerankerClient.from_env()
    store = PostgresChunkStore.connect()
    conversation_id = args.conversation_id or uuid4().hex
    try:
        store.migrate()
        if args.conversation_id is None:
            store.create_conversation(conversation_id, args.user_id)
        history = store.conversation_history(conversation_id, args.user_id, args.history)
        search_query = rewrite_query(chat, history, args.query)
        embedder = index.embedder_spec().create()
        query_vector = embedder.embed_query(search_query)
        candidates = index.hybrid_search(
            search_query,
            query_vector,
            k=args.candidate_k,
            filters={"role": args.role, "document": args.document, "doc_type": args.doc_type},
        )
        permitted = filter_authorized_hits(candidates, store, args.role, folder.name)
        hits = reranker.rerank(search_query, permitted, index.texts, top_n=args.k)
        answer = generate_grounded_answer(
            chat, args.query, search_query, hits, index.texts, history
        )
        store.save_turn(conversation_id, args.user_id, args.query, answer)
    finally:
        store.close()

    print(f"Conversation: {conversation_id}")
    print(answer)
    if hits:
        print("Sources:")
        for hit in hits:
            print(f"- [{hit.chunk_id}] p.{hit.pages} {hit.meta.get('section') or '-'}")


def register(add, paths: Paths) -> None:
    p = add("ask", run_ask, "Retrieve verified, role-authorized evidence and generate a grounded answer.")
    p.add_argument("query", help="Question to answer.")
    p.add_argument("--user-id", required=True, help="Stable caller ID used to scope conversation history.")
    p.add_argument("--role", required=True, help="Caller role; only verified documents for this role are eligible.")
    p.add_argument("--conversation-id", help="Continue an existing conversation owned by --user-id.")
    p.add_argument("-k", type=positive_int, default=5, help="Number of reranked chunks passed to answer generation.")
    p.add_argument("--candidate-k", type=positive_int, default=40, help="Dense/BM25 candidate pool size.")
    p.add_argument("--history", type=positive_int, default=10, help="Maximum prior messages used for query rewriting.")
    p.add_argument("--document", help="Restrict retrieval to this document stem.")
    p.add_argument("--doc-type", choices=["pdf", "docx"], help="Restrict retrieval to this source type.")
    p.add_argument("--index", help="Index folder (default: newest under outputs/index).")
    p.add_argument("--ingest", default=paths.relative(paths.ingest), help="Ingest folder for the index staleness check.")
