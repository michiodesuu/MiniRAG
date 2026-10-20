"""Facts about each candidate embedding model, read from its Hugging Face model
card and repository files on 2026-10-19 (not from memory). The revision pins the
exact commit; prefixes and pooling are what the card / sentence-transformers
config require. Parameter counts: from the repository's safetensors metadata,
or estimated as weight-file bytes / 4 (float32) where marked "~".
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelProfile:
    revision: str              # commit hash on the Hugging Face hub
    license: str
    parameters: str
    dim: int
    max_input: int             # the model's own maximum (position embeddings)
    max_seq_length: int        # what we set explicitly (cost grows with length)
    pooling: str               # from 1_Pooling/config.json
    query_prefix: str = ""
    passage_prefix: str = ""
    note: str = ""


PROFILES = {
    "BAAI/bge-m3": ModelProfile(
        "5617a9f61b028005a4858fdac845db406aefb181", "MIT", "~568M (2.27 GB float32)",
        1024, 8192, 1024, "cls",
        note="XLM-RoBERTa large; card: 'no longer requires adding instructions to the queries'"),
    "intfloat/multilingual-e5-base": ModelProfile(
        "d128750597153bb5987e10b1c3493a34e5a4502a", "MIT", "278M", 768, 512, 512, "mean",
        "query: ", "passage: ",
        note="XLM-RoBERTa base; card: 'Each input text should start with \"query: \" or \"passage: \"'"),
    "intfloat/multilingual-e5-small": ModelProfile(
        "614241f622f53c4eeff9890bdc4f31cfecc418b3", "MIT", "118M", 384, 512, 512, "mean",
        "query: ", "passage: ", note="12-layer BERT; same prefixes as e5-base"),
    "airesearch/WangchanX-Legal-ThaiCCL-Retriever": ModelProfile(
        "2a0337195e9ceedab71f48c18971a6db55caf0eb", "MIT", "568M", 1024, 8192, 1024, "cls",
        note="bge-m3 fine-tuned on Thai corporate/commercial law; card gives no usage or "
             "prefix, so bge-m3's (none) is assumed"),
    "kornwtp/simcse-model-m-bert-thai-cased": ModelProfile(
        "10ca1e5f966ec2159704b260a128fad3a05c0096", "Apache-2.0", "~178M (711 MB float32)",
        768, 512, 512, "cls", note="mBERT + unsupervised SimCSE on Thai Wikipedia"),
}

DEFAULT_BATCH_SIZE = 8  # passages per forward pass on CPU

# Used when --model is not given; chosen from the retrieval comparison (note.md 2.17):
# tied best with WangchanX on 30 questions, general-purpose, best role accuracy.
# Low-RAM fallback: intfloat/multilingual-e5-small (half the memory, ~9x faster queries).
DEFAULT_MODEL = "BAAI/bge-m3"

# Considered and left out: kornwtp/ConGen-simcse-model-roberta-base-thai (card says
# Apache-2.0 but does not name its base model, so the licence chain cannot be checked).


def profile(model: str) -> ModelProfile | None:
    return PROFILES.get(model)
