"""Embedders, cache and the chunk command: no model, no network."""
import json

import pytest

np = pytest.importorskip("numpy")
from rag_ingest.cli import main  # noqa: E402
from rag_ingest.embeddings import create_embedder  # noqa: E402
from rag_ingest.embeddings.cache import EmbeddingCache, cache_key, embed_with_cache  # noqa: E402


def test_fake_embedder_is_deterministic_and_normalised():
    e = create_embedder("fake")
    a = e.embed_passages(["ส่งคำขอแก้ไขไปที่สำนักงานใหญ่", "ออกรายงานอุบัติเหตุ"])
    b = create_embedder("fake").embed_passages(["ส่งคำขอแก้ไขไปที่สำนักงานใหญ่", "ออกรายงานอุบัติเหตุ"])
    assert a.dtype == np.float32 and a.shape == (2, 64)
    assert np.array_equal(a, b)                       # same in every run (blake2b, not hash())
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)
    query = e.embed_query("ส่งคำขอแก้ไข")
    assert float(a[0] @ query) > float(a[1] @ query)  # shared n-grams score higher


def test_unknown_embedder():
    with pytest.raises(ValueError, match="Unknown embedder"):
        create_embedder("nope")


def test_cache_key_changes_with_model_revision_settings_and_text():
    settings = {"model": "m", "revision": "r1", "dim": 4, "max_len": 512}
    key = cache_key(settings, "ข้อความ")
    assert key == cache_key(dict(settings), "ข้อความ")
    for change in ({"model": "m2"}, {"revision": "r2"}, {"max_len": 256}):
        assert cache_key({**settings, **change}, "ข้อความ") != key
    assert cache_key(settings, "ข้อความอื่น") != key


def test_cache_embeds_only_new_texts(tmp_path):
    e, cache = create_embedder("fake"), EmbeddingCache(tmp_path / "cache")
    first = embed_with_cache(e, ["ก", "ข"], cache)
    second = embed_with_cache(e, ["ก", "ข", "ค"], cache)
    assert (first.computed, second.computed) == (2, 1)
    assert np.array_equal(first.vectors, second.vectors[:2])


def test_local_snapshot_downloads_one_weights_file(monkeypatch, tmp_path):
    import huggingface_hub
    from rag_ingest.embeddings import local

    calls = {}
    monkeypatch.setattr(huggingface_hub, "list_repo_files",
                        lambda model, revision: ["model.safetensors", "pytorch_model.bin", "onnx/model.onnx"])
    def fake_download(model, revision=None, allow_patterns=None, local_files_only=False):
        calls.update(allow_patterns=allow_patterns)
        return str(tmp_path / "abc123")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_download)
    path = local.snapshot("org/model", "abc123")
    assert path.name == "abc123"
    assert "model.safetensors" in calls["allow_patterns"]
    assert "pytorch_model.bin" not in calls["allow_patterns"] and "onnx/model.onnx" not in calls["allow_patterns"]


INGEST = """<!-- page 1 -->
1. เริ่มต้น

ระบบใช้บันทึกอุบัติเหตุของสาขา
"""


def write_ingest(folder, extra=""):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "คู่มือ.md").write_text(INGEST + extra, encoding="utf-8")


@pytest.fixture
def project_root(tmp_path, monkeypatch):
    """A temporary project root, so the command does not scan the real data/ folder."""
    monkeypatch.setenv("RAG_INGEST_ROOT", str(tmp_path))
    return tmp_path


def test_chunk_command_with_fake_embedder(tmp_path, project_root, capsys):
    write_ingest(tmp_path / "ingest")
    main(["chunk", "--embedder", "fake", "--input", str(tmp_path / "ingest"), "--output", str(tmp_path / "chunks")])
    assert "Chunks: 1 from 1 document(s)" in capsys.readouterr().out
    manifest = json.loads((tmp_path / "chunks" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["settings"]["tokenizer"]["token_counter"] == "approximate"
    assert manifest["exclusion_fingerprints"]["images"] >= 0


def test_chunk_command_stops_on_personal_data(tmp_path, project_root):
    base = "310170012345"
    good = base + str((11 - sum(int(d) * (13 - i) for i, d in enumerate(base)) % 11) % 10)
    write_ingest(tmp_path / "ingest", f"\nเลขบัตร {good}\n")
    args = ["chunk", "--embedder", "fake", "--input", str(tmp_path / "ingest"), "--output", str(tmp_path / "chunks")]
    with pytest.raises(SystemExit, match="personal data found"):
        main(args)
    main(args + ["--allow-pii"])  # finishes, but the chunk still stays out
    assert good not in (tmp_path / "chunks" / "chunks.jsonl").read_text(encoding="utf-8")


class FakeResponse:
    status_code = 200

    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


def test_api_embedder_request_order_key_and_settings(monkeypatch, capsys):
    httpx = pytest.importorskip("httpx")
    from rag_ingest.embeddings.api import ApiEmbedder

    sent = []
    def fake_post(url, json, headers, timeout):
        sent.append((url, json, headers))
        rows = [{"index": i, "embedding": [float(i + 1), 0.0, 1.0]} for i in range(len(json["input"]))]
        return FakeResponse({"data": list(reversed(rows))})  # out of order on purpose
    monkeypatch.setattr(httpx, "post", fake_post)
    e = ApiEmbedder("http://embed.internal/v1/embeddings", "intfloat/multilingual-e5-small", key="secret-key")
    assert "unencrypted" in capsys.readouterr().err
    vectors = e.embed_passages(["ก", "ข"])
    assert sent[0][1] == {"model": "intfloat/multilingual-e5-small", "input": ["passage: ก", "passage: ข"]}
    assert sent[0][2] == {"Authorization": "Bearer secret-key"}
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0) and vectors[0][0] < vectors[1][0]  # order restored
    e.embed_query("ค")
    assert sent[-1][1]["input"] == ["query: ค"]
    settings = e.settings()
    assert "secret-key" not in json.dumps(settings) and settings["endpoint"] == "embed.internal"
    assert settings["token_counter"] == "approximate"


def test_api_options_need_url_and_model(monkeypatch):
    import argparse
    from rag_ingest.embeddings.api import api_options

    for name in ("EMBED_API_URL", "EMBED_API_MODEL", "EMBED_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    args = argparse.Namespace(api_url=None, api_model=None, tokenizer=None)
    with pytest.raises(SystemExit, match="needs --api-url"):
        api_options(args)
    monkeypatch.setenv("EMBED_API_URL", "https://embed.internal/v1/embeddings")
    monkeypatch.setenv("EMBED_API_MODEL", "m")
    monkeypatch.setenv("EMBED_API_KEY", "k")
    options = api_options(args)
    assert (options["url"], options["model"], options["key"]) == ("https://embed.internal/v1/embeddings", "m", "k")


def test_api_never_exposes_url_secrets(monkeypatch):
    httpx = pytest.importorskip("httpx")
    from rag_ingest.embeddings.api import ApiEmbedder

    class Failing(FakeResponse):
        status_code = 401
    monkeypatch.setattr(httpx, "post", lambda *a, **k: Failing({}))
    e = ApiEmbedder("https://user:pa55word@embed.internal:8443/v1/embeddings?api_key=s3cret", "m")
    assert e.settings()["endpoint"] == "embed.internal:8443"
    with pytest.raises(RuntimeError) as error:
        e.embed_passages(["ก"])
    message = str(error.value)
    assert "pa55word" not in message and "s3cret" not in message and "embed.internal:8443" in message
