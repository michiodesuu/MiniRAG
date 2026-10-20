import json

import pytest

from rag_ingest import benchmark as bm
from rag_ingest.ground_truth import BLANK_BOX, CLAUDE, HUMAN, GroundTruthStore, benchmark_images
from rag_ingest.models import Recognition
from rag_ingest.review import BlankSource
from rag_ingest.stores import VisionStore
from rag_ingest.textproc import NO_TEXT_MARK

SCREEN, ICON = "doc_page_1_image_1", "doc_page_1_image_2"


@pytest.fixture
def truth(paths):
    return GroundTruthStore(paths.ground_truth)


@pytest.fixture
def manifest(paths):
    paths.ingest.mkdir(parents=True)
    path = paths.ingest / "manifest.json"
    path.write_text(json.dumps([{"image": SCREEN, "status": "ocr"}, {"image": ICON, "status": "icon"},
                                {"image": "dup", "status": "duplicate"}]), encoding="utf-8")
    return path


def test_save_writes_text_and_provenance(truth):
    truth.save(SCREEN, "  ข้อความ \n", CLAUDE, BLANK_BOX, note="unchecked")
    assert truth.texts() == {SCREEN: "ข้อความ\n"}
    assert truth.provenance()[SCREEN]["by"] == CLAUDE
    assert truth.provenance()[SCREEN]["note"] == "unchecked"


def test_save_cannot_leave_the_folder(truth):
    truth.save("../../escape", "x", HUMAN, BLANK_BOX)
    assert truth.has("escape")


def test_benchmark_images_skip_icons_and_duplicates(manifest):
    assert benchmark_images(manifest) == [SCREEN]


def test_benchmark_images_without_manifest(tmp_path):
    assert benchmark_images(tmp_path / "missing.json") is None


@pytest.fixture
def blank(truth, make_image):
    return BlankSource(truth, {SCREEN: make_image(f"{SCREEN}.png")})


def test_blank_box_shows_no_model_text(blank):
    assert blank.items() == [{"name": SCREEN, "text": "", "corrected_text": None, "status": None}]


def test_blank_box_saves_human_ground_truth(blank, truth):
    blank.save(SCREEN, "พิมพ์เอง")
    assert truth.provenance()[SCREEN]["method"] == BLANK_BOX
    assert blank.items()[0]["corrected_text"] == "พิมพ์เอง"


def test_blank_box_hides_claude_transcriptions(blank, truth):
    truth.save(SCREEN, "อ่านโดย Claude", CLAUDE, BLANK_BOX)
    assert blank.items()[0]["corrected_text"] is None


def test_blank_box_knows_only_its_images(blank):
    assert not blank.has(ICON)


@pytest.fixture
def corrected(paths, make_image):
    store = VisionStore(paths.method("vision"), paths)
    store.save(make_image(f"{SCREEN}.png"), Recognition("model text"))
    store.save_correction(SCREEN, "corrected text")
    return store


def test_correction_counts_as_human_edit(corrected, truth):
    loaded = bm.load_ground_truth([corrected], truth)
    assert (loaded[SCREEN].by, loaded[SCREEN].method) == (HUMAN, "edited from vision model text")


def test_ground_truth_file_wins_over_correction(corrected, truth):
    truth.save(SCREEN, "blank box text", CLAUDE, BLANK_BOX)
    assert bm.load_ground_truth([corrected], truth)[SCREEN].by == CLAUDE


def test_ground_truth_limited_to_benchmark_images(corrected, truth):
    assert bm.load_ground_truth([corrected], truth, images=[ICON]) == {}


def test_failures_count_rejections_and_missing():
    failures = bm.count_failures({"a": NO_TEXT_MARK, "b": "ข้อความจริง"}, ["a", "b", "c"])
    assert (failures.rejected, failures.missing, failures.total) == ({"no_text_marker": 1}, 1, 2)


def test_pipeline_falls_back_when_first_source_is_rejected():
    outputs = {"first": {"a": NO_TEXT_MARK, "b": "แรก"}, "second": {"a": "สำรอง", "b": "สอง"}}
    assert bm.pipeline_outputs(outputs, ("first", "second", "absent")) == {"a": "สำรอง", "b": "แรก"}


def test_extra_vision_folders_are_discovered(paths):
    extra = paths.outputs / "vision_newprompt"
    extra.mkdir(parents=True)
    (extra / "x.json").write_text("{}", encoding="utf-8")
    names = [m.name for m in bm.discover_methods(paths, exclude=[])]
    assert "vision_newprompt" in names


def test_run_without_ground_truth_returns_none(paths):
    assert bm.run(paths, paths.ground_truth, []) is None


def test_report_splits_by_provenance(paths, corrected, truth):
    result = bm.run(paths, paths.ground_truth, [])
    assert "## Ground truth by a person (n=1)" in result.report
