import json
import shutil
from pathlib import Path

import pytest

from rag_ingest.ocr_eval.benchmark import Truth, render, score_all, summarize
from rag_ingest.ocr_eval.compare import hint_label
from rag_ingest.ingest import Ingestor, OcrCache, OcrSourceChain, run_ingest
from rag_ingest.models import ImageRef
from rag_ingest.models import EmbeddedImage, Page, TextBlock
from rag_ingest.stores import VisionStore
from rag_ingest.textproc import NO_TEXT_MARK

ROOT = Path(__file__).resolve().parents[1]


def image(index, data, size=(800, 400), pos=(100.0, 50.0)):
    return EmbeddedImage(ImageRef("doc", 1, index), data, "png", *size, *pos)


def vision_folder(paths, name, texts):
    store = VisionStore(paths.method(name), paths)
    for stem, text in texts.items():
        store.write(stem, {"image": "x", "text": text})
    return store


def make_ingestor(paths, tmp_path, sources):
    stores = [vision_folder(paths, name, texts) for name, texts in sources.items()]
    return Ingestor(OcrSourceChain(stores), OcrCache(tmp_path / "cache.json"))


def test_page_merges_text_and_screenshots_by_position(paths, tmp_path):
    ingestor = make_ingestor(paths, tmp_path, {"good": {"doc_page_1_image_1": "เมนูหลัก"}})
    page = Page(1, 800.0, "", [TextBlock(300, 0, "หลังภาพ"), TextBlock(10, 0, "หัวข้อ")],
                [image(1, b"a", pos=(100, 50))])
    assert ingestor.page_text(page) == (
        "<!-- page 1 -->\nหัวข้อ\n\n[ภาพหน้าจอ]\nเมนูหลัก\n[/ภาพหน้าจอ]\n\nหลังภาพ")


def test_icons_skipped_duplicates_reuse_text(paths, tmp_path):
    ingestor = make_ingestor(paths, tmp_path, {"good": {"doc_page_1_image_1": "เมนูหลัก"}})
    page = Page(1, 800.0, "", [], [
        image(1, b"same", pos=(10, 0)),
        image(2, b"icon", size=(40, 40)),
        image(3, b"same", pos=(500, 0)),  # same bytes as image 1, no result of its own
    ])
    text = ingestor.page_text(page)
    assert text.count("เมนูหลัก") == 2
    statuses = [(e["image"], e["status"]) for e in ingestor.result.manifest]
    assert statuses == [("doc_page_1_image_1", "ocr"), ("doc_page_1_image_2", "icon"),
                        ("doc_page_1_image_3", "duplicate")]
    assert ingestor.result.manifest[2]["duplicate_of"] == "doc_page_1_image_1"


def test_bad_ocr_falls_back_and_records_reason(paths, tmp_path):
    ingestor = make_ingestor(paths, tmp_path, {
        "primary": {"doc_page_1_image_1": NO_TEXT_MARK},
        "fallback": {"doc_page_1_image_1": "สำรอง"},
    })
    ingestor.page_text(Page(1, 800.0, "", [], [image(1, b"a")]))
    entry = ingestor.result.manifest[0]
    assert entry["source"] == "fallback"
    assert entry["rejected"] == [{"source": "primary", "reason": "no_text_marker"}]


def test_missing_ocr_is_marked_not_dropped(paths, tmp_path):
    ingestor = make_ingestor(paths, tmp_path, {"primary": {}})
    text = ingestor.page_text(Page(1, 800.0, "", [], [image(1, b"a"), image(2, b"a")]))
    assert text.count("[ภาพหน้าจอ: ยังไม่ได้ OCR (doc_page_1_image_") == 2
    assert ingestor.result.needs_ocr == ["doc_page_1_image_1"]  # duplicates not listed twice
    assert [e["status"] for e in ingestor.result.manifest] == ["needs_ocr", "duplicate"]


def test_undecodable_image_is_reported_not_ocrd(paths, tmp_path):
    ingestor = make_ingestor(paths, tmp_path, {"good": {"doc_page_1_image_1": "ไม่ควรใช้"}})
    text = ingestor.page_text(Page(1, 800.0, "", [], [image(1, b"emf", size=(0, 0))]))
    assert "ไม่ควรใช้" not in text and "ภาพหน้าจอ" not in text
    assert ingestor.result.manifest[0]["status"] == "unsupported_format"
    assert ingestor.result.needs_ocr == []


def test_unplaced_image_goes_to_page_end(paths, tmp_path):
    ingestor = make_ingestor(paths, tmp_path, {"good": {"doc_page_1_image_1": "ท้าย"}})
    page = Page(1, 800.0, "", [TextBlock(700, 0, "ล่างสุด")], [image(1, b"a", pos=(None, None))])
    assert ingestor.page_text(page).endswith("ล่างสุด\n\n[ภาพหน้าจอ]\nท้าย\n[/ภาพหน้าจอ]")


def test_cache_drops_unusable_entries(tmp_path):
    path = tmp_path / "cache.json"
    path.write_text(json.dumps({"a": {"text": "ok", "source": "s", "image": "i"},
                                "b": {"text": NO_TEXT_MARK, "source": "s", "image": "i"}}),
                    encoding="utf-8")
    assert list(OcrCache(path).entries) == ["a"]


SAMPLE_PDFS = [ROOT / "data" / "raw" / "pdf" / f"{name}.pdf" for name in ("ผู้จัดการสาขา", "ผู้จัดการเขต")]


@pytest.mark.skipif(not all(p.exists() for p in SAMPLE_PDFS), reason="sample PDFs not present")
def test_real_pdfs_end_to_end(tmp_path):
    docs = tmp_path / "docs"  # only the two known manuals, whatever else is in data/raw
    docs.mkdir()
    for pdf in SAMPLE_PDFS:
        shutil.copy(pdf, docs / pdf.name)
    result = run_ingest(docs, tmp_path / "out", stores=[])
    assert (result.count("icon"), result.count("duplicate"), len(result.unique)) == (7, 2, 18)
    text = (tmp_path / "out" / "ผู้จัดการสาขา.md").read_text(encoding="utf-8")
    assert "ส่งคำขอแก้ไขและลบ" in text and "ท าการ" not in text


def test_benchmark_scores_and_summary():
    truth = {stem: Truth(text, "human", "test") for stem, text in {"a": "abcd", "b": "xyz"}.items()}
    methods = {"m1": {"a": "abcd", "b": "xyq"}, "m2": {"a": ""}}
    rows = score_all(truth, methods)
    assert rows[0].scores["m1"].cer == pytest.approx(0)
    assert rows[0].scores["m2"].cer == pytest.approx(1)
    stats = summarize(rows, ["m1", "m2"])
    assert stats["m1"] == (pytest.approx(1 / 6), pytest.approx(1 / 7), 0, 2)
    assert stats["m2"] == (pytest.approx(1), pytest.approx(1), 0, 1)
    assert "| m1 | 0.167 |" in render(rows, ["m1", "m2"])


def test_hint_label():
    assert hint_label({"settings": {"paddle_hint": True, "tesseract_hint": True}}) == "P+T"
    assert hint_label({"settings": {"tesseract_hint": True}}) == "T"
    assert hint_label({}) == "-"
