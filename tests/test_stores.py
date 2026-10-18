import json
import math

from rag_ingest.models import OcrLine, Recognition
from rag_ingest.stores import LineStore, VisionStore, open_store


def test_line_store_round_trip(paths, make_image):
    store = LineStore(paths.method("tesseract"))
    image = make_image("a_page_1_image_1.png")
    store.save(image, Recognition("x", [OcrLine("บันทึก", 0.91234), OcrLine("ok", 0.5)]))

    assert store.path("a_page_1_image_1").read_text(encoding="utf-8") == \
        "บันทึก\tconfidence=0.9123\nok\tconfidence=0.5000\n"
    assert store.text("a_page_1_image_1") == "บันทึก\nok"
    assert store.confident_lines("a_page_1_image_1", 0.8) == ["บันทึก"]


def test_line_store_tolerates_bad_lines(paths):
    store = LineStore(paths.method("ocr"))
    store.folder.mkdir()
    store.path("img").write_text("no confidence\nbad\tconfidence=x\n\tconfidence=0.9\n",
                                 encoding="utf-8")
    lines = store.lines("img")
    assert [l.text for l in lines] == ["no confidence", "bad", ""]
    assert math.isnan(lines[0].confidence) and math.isnan(lines[1].confidence)
    # Unparsable confidence and empty text never become hints; text is kept for scoring.
    assert store.confident_lines("img", 0.0) == []
    assert store.text("img") == "no confidence\nbad\n"
    assert store.lines("missing") == []


def test_vision_store_save_and_correct(paths, make_image):
    store = VisionStore(paths.method("vision"), paths)
    image = make_image("a_page_1_image_1.png")
    store.save(image, Recognition("model text", model="m", settings={"temperature": 0}))

    record = store.load("a_page_1_image_1")
    assert record == {
        "image": "outputs/images/a_page_1_image_1.png",
        "model": "m",
        "prompt_file": "prompts/thai_document_ocr.txt",
        "settings": {"temperature": 0},
        "text": "model text",
        "review_status": "pending",
    }
    assert store.image_path(record) == image

    store.save_correction("a_page_1_image_1", " fixed \n")
    assert store.corrections() == {"a_page_1_image_1": "fixed"}
    assert store.final_texts() == {"a_page_1_image_1": "fixed"}
    assert store.load("a_page_1_image_1")["review_status"] == "corrected"


def test_vision_store_reads_windows_paths(paths, make_image):
    image = make_image("b_page_2_image_1.png")
    store = VisionStore(paths.method("vision"), paths)
    assert store.image_path({"image": "outputs\\images\\b_page_2_image_1.png"}) == image


def test_store_paths_cannot_escape_folder(paths):
    store = VisionStore(paths.method("vision"), paths)
    assert store.path("../../secret").parent == store.folder


def test_open_store_factory(paths):
    assert isinstance(open_store(paths, "vision_hint"), VisionStore)
    assert isinstance(open_store(paths, "tesseract"), LineStore)
    assert open_store(paths, "tesseract").folder == paths.method("tesseract")

    unknown = paths.outputs / "my_run"
    unknown.mkdir()
    (unknown / "x.txt").write_text("a\tconfidence=1\n", encoding="utf-8")
    assert isinstance(open_store(paths, "outputs/my_run"), LineStore)
    (unknown / "y.json").write_text(json.dumps({"text": ""}), encoding="utf-8")
    assert isinstance(open_store(paths, "outputs/my_run"), VisionStore)
