from pathlib import Path

import pytest
from PIL import Image

from rag_ingest.config import Paths


@pytest.fixture
def paths(tmp_path: Path) -> Paths:
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "thai_document_ocr.txt").write_text("PROMPT", encoding="utf-8")
    (tmp_path / "outputs" / "images").mkdir(parents=True)
    return Paths(tmp_path)


@pytest.fixture
def make_image(paths: Paths):
    def make(name: str, size=(200, 100)) -> Path:
        path = paths.images / name
        Image.new("RGB", size, "white").save(path)
        return path
    return make
