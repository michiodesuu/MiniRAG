"""sentence-transformers on CPU, with the model revision pinned.

How a text becomes a vector here (also in note.md):
1. prefix: e5 models need "passage: " / "query: " (profiles.py); others none
2. tokenise with the model's own tokenizer (SentencePiece for XLM-R models),
    add special tokens, truncate at max_seq_length
3. the transformer encoder gives one vector per token
4. pooling: CLS token or mean of token vectors, as the model's
    1_Pooling/config.json says
5. Normalize layer: the vector is scaled to length 1 (checked again here)

The tokenizer is loaded on its own first, so `chunk` can count tokens without
loading the model weights; the model loads on the first embed call.
"""
from importlib.metadata import version
from pathlib import Path
import json

import numpy as np

from rag_ingest.embeddings.vectors import normalise_rows
from rag_ingest.embeddings.profiles import profile

from rag_ingest.embeddings.profiles import DEFAULT_BATCH_SIZE  # noqa: E402
# Only what sentence-transformers needs; repositories also hold ONNX/OpenVINO
# copies of the weights that would multiply the download.
MODEL_FILES = ["config.json", "config_sentence_transformers.json", "modules.json",
               "sentence_bert_config.json", "tokenizer.json", "tokenizer_config.json",
               "special_tokens_map.json", "sentencepiece.bpe.model", "vocab.txt",
               "1_Pooling/config.json"]
WEIGHTS = ("model.safetensors", "pytorch_model.bin")  # first one the repository has
COUNTING_MAX_LENGTH = 1_000_000


def snapshot(model: str, revision: str | None) -> Path:
    """Local copy of the model at an exact commit: downloaded once (config, tokenizer
    and one weights file), then used offline."""
    from huggingface_hub import list_repo_files, snapshot_download
    from huggingface_hub.utils import disable_progress_bars

    disable_progress_bars()
    try:
        files = set(list_repo_files(model, revision=revision))
    except Exception:  # offline: use what is already cached
        return Path(snapshot_download(model, revision=revision, local_files_only=True))
    weights = next(w for w in WEIGHTS if w in files)
    return Path(snapshot_download(model, revision=revision, allow_patterns=MODEL_FILES + [weights]))


class LocalEmbedder:
    def __init__(self, model: str, revision: str | None = None,
                 batch_size: int = DEFAULT_BATCH_SIZE, max_seq_length: int | None = None):
        known = profile(model)
        self.model = model
        self.path = snapshot(model, revision or (known.revision if known else None))
        self.revision = self.path.name  # snapshot folders are named by commit hash
        self.batch_size = batch_size
        self.query_prefix = known.query_prefix if known else ""
        self.passage_prefix = known.passage_prefix if known else ""
        self.max_seq_length = max_seq_length or (known.max_seq_length if known else 512)
        self._tokenizer = None
        self._encoder = None
        self.dim = known.dim if known else 0

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(str(self.path))
            # Counting must see the full length of a passage before it is split, so the
            # length warning does not apply here (encoding truncates separately).
            self._tokenizer.model_max_length = COUNTING_MAX_LENGTH
        return self._tokenizer

    @property
    def encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(str(self.path), device="cpu")
            self._encoder.max_seq_length = self.max_seq_length
            self.dim = self._encoder.get_embedding_dimension()
        return self._encoder

    def count_tokens(self, text: str) -> int:
        """Tokens of the text as a passage, with prefix and special tokens, before truncation."""
        return len(self.tokenizer(self.passage_prefix + text, add_special_tokens=True,
                                truncation=False)["input_ids"])

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.encoder.encode(texts, batch_size=self.batch_size, convert_to_numpy=True,
                                    normalize_embeddings=True, show_progress_bar=False)
        return normalise_rows(vectors)

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return self._encode([self.passage_prefix + t for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([self.query_prefix + text])[0]

    def pooling(self) -> str:
        known = profile(self.model)
        config = self.path / "1_Pooling" / "config.json"
        if config.exists():
            modes = json.loads(config.read_text(encoding="utf-8"))
            return next((k.removeprefix("pooling_mode_").removesuffix("_tokens").removesuffix("_token")
                        for k, v in modes.items() if k.startswith("pooling_mode_") and v is True), "?")
        return known.pooling if known else "?"

    def settings(self) -> dict:
        return {"model": self.model, "revision": self.revision, "dim": self.dim,
                "pooling": self.pooling(), "normalise": True, "max_len": self.max_seq_length,
                "query_prefix": self.query_prefix, "passage_prefix": self.passage_prefix,
                "token_counter": "model tokenizer",
                "library": {"sentence-transformers": version("sentence-transformers"),
                            "transformers": version("transformers"), "torch": version("torch")}}
