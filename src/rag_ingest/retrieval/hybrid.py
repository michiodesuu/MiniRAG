"""BM25 (sparse keyword search) and Reciprocal Rank Fusion (RRF) for hybrid search (W08)."""
from collections import Counter
import math
import re
from typing import Callable, Iterable

from rag_ingest.retrieval.vectorstore import FILTER_FIELDS, Hit

# Regex-based word tokenizer fallback for Thai and alphanumeric characters
THAI_WORD_RE = re.compile(r"[\u0e00-\u0e7f]+|[a-zA-Z0-9]+", re.UNICODE)


def default_thai_tokenizer(text: str) -> list[str]:
    """Tokenize Thai and English text.
    Uses pythainlp if installed; otherwise falls back to character-ngram + word tokens
    for robust Thai keyword matching without external dependencies.
    """
    try:
        from pythainlp.tokenize import word_tokenize
        return [w.strip().lower() for w in word_tokenize(text, engine="newmm") if w.strip()]
    except ImportError:
        tokens: list[str] = []
        for match in THAI_WORD_RE.findall(text):
            w = match.lower()
            tokens.append(w)
            # For Thai runs without space separation, include character 3-grams so sub-words match
            if "\u0e00" <= w[0] <= "\u0e7f" and len(w) > 3:
                tokens.extend(w[i : i + 3] for i in range(len(w) - 2))
        return tokens


class BM25Index:
    """Okapi BM25 index over in-memory text chunks."""

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: Callable[[str], list[str]] = default_thai_tokenizer,
    ):
        self.k1 = k1
        self.b = b
        self.tokenizer = tokenizer
        self.corpus_size = 0
        self.avg_dl = 0.0
        self.doc_lengths: list[int] = []
        self.doc_ids: list[str] = []
        self.doc_metas: list[dict] = []
        self.doc_freqs: dict[str, int] = {}
        self.term_freqs: list[Counter[str]] = []

    def fit(self, documents: Iterable[tuple[str, str, dict]]) -> "BM25Index":
        """Index a collection of (chunk_id, text, meta)."""
        self.doc_ids = []
        self.doc_metas = []
        self.doc_lengths = []
        self.term_freqs = []
        self.doc_freqs = Counter()

        total_length = 0
        for chunk_id, text, meta in documents:
            tokens = self.tokenizer(text)
            dl = len(tokens)
            tf = Counter(tokens)
            self.doc_ids.append(chunk_id)
            self.doc_metas.append(meta)
            self.doc_lengths.append(dl)
            self.term_freqs.append(tf)
            total_length += dl
            for term in tf.keys():
                self.doc_freqs[term] += 1

        self.corpus_size = len(self.doc_ids)
        self.avg_dl = (total_length / self.corpus_size) if self.corpus_size > 0 else 0.0
        return self

    def _idf(self, term: str) -> float:
        df = self.doc_freqs.get(term, 0)
        return math.log(1.0 + (self.corpus_size - df + 0.5) / (df + 0.5))

    def _check_filters(self, filters: dict | None) -> None:
        for field in (filters or {}):
            if field not in FILTER_FIELDS:
                raise ValueError(f"Unknown filter {field!r}; use {', '.join(FILTER_FIELDS)}")

    def search(self, query: str, k: int = 10, filters: dict | None = None) -> list[Hit]:
        """Search query using Okapi BM25 with optional metadata filtering."""
        self._check_filters(filters)
        if self.corpus_size == 0 or k <= 0:
            return []

        tokens = self.tokenizer(query)
        if not tokens:
            return []

        scores: list[float] = [0.0] * self.corpus_size
        idfs = {term: self._idf(term) for term in set(tokens)}

        for term in set(tokens):
            idf = idfs[term]
            if idf <= 0:
                continue
            for idx, tf_dict in enumerate(self.term_freqs):
                tf = tf_dict.get(term, 0)
                if tf == 0:
                    continue
                dl = self.doc_lengths[idx]
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (dl / (self.avg_dl or 1.0)))
                scores[idx] += idf * (tf * (self.k1 + 1.0)) / denominator

        results: list[Hit] = []
        for idx, score in enumerate(scores):
            if score <= 0.0:
                continue
            meta = self.doc_metas[idx]
            if filters:
                if any(meta.get(key) != val for key, val in filters.items() if val is not None):
                    continue
            results.append(Hit(chunk_id=self.doc_ids[idx], score=float(score), meta=meta))

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:k]


def reciprocal_rank_fusion(
    ranked_lists: list[list[Hit]],
    k_rrf: int = 60,
    top_k: int = 10,
) -> list[Hit]:
    """Combine multiple ranked Hit lists using Reciprocal Rank Fusion (RRF).

    RRF_score(doc) = SUM( 1 / (k_rrf + rank_i(doc)) ) for each ranked list i.
    """
    if top_k <= 0:
        return []
    rrf_scores: dict[str, float] = Counter()
    hit_map: dict[str, Hit] = {}

    for ranked in ranked_lists:
        for rank, hit in enumerate(ranked, start=1):
            chunk_id = hit.chunk_id
            rrf_scores[chunk_id] += 1.0 / (k_rrf + rank)
            if chunk_id not in hit_map:
                hit_map[chunk_id] = hit

    sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

    return [
        Hit(chunk_id=cid, score=float(rrf_scores[cid]), meta=hit_map[cid].meta)
        for cid in sorted_doc_ids[:top_k]
    ]
