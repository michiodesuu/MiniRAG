import pytest

from rag_ingest.cli import build_parser, engine_options, warn_if_clear_text


@pytest.mark.parametrize("url, warns", [
    ("http://ocr.example.local/v1/chat/completions", True),  # NOSONAR test input for the clear-text warning
    ("https://ocr.example.com/v1/chat/completions", False),
    ("http://localhost:8000/v1/chat/completions", False),  # NOSONAR test input for the clear-text warning
    ("http://127.0.0.1:8000/v1/chat/completions", False),  # NOSONAR test input for the clear-text warning
])
def test_clear_text_warning(url, warns, capsys):
    warn_if_clear_text(url)
    assert ("WARNING" in capsys.readouterr().err) == warns


def parse(paths, *argv):
    return build_parser(paths).parse_args(["ocr", *argv])


def test_engine_options(paths, monkeypatch):
    monkeypatch.setenv("OCR_API_KEY", "k")
    assert engine_options(parse(paths, "--engine", "tesseract", "--psm", "6"), paths) == \
        {"upscale": 0, "psm": 6}
    assert engine_options(parse(paths, "--engine", "paddle"), paths) == {}
    ollama = engine_options(parse(paths, "--engine", "ollama", "--hint", "both"), paths)
    assert ollama["hint_source"] == "both" and ollama["model"] == "qwen2.5vl:3b"
    api = engine_options(parse(paths, "--engine", "api", "--api-url", "https://x/v1",
                               "--api-model", "m"), paths)
    assert (api["url"], api["model"], api["key"]) == ("https://x/v1", "m", "k")


def test_engine_options_rejects_bad_combinations(paths):
    with pytest.raises(SystemExit, match="--hint only applies"):
        engine_options(parse(paths, "--engine", "tesseract", "--hint"), paths)
    with pytest.raises(SystemExit, match="needs --api-url"):
        engine_options(parse(paths, "--engine", "api", "--api-url", "", "--api-model", ""), paths)
