import subprocess
import sys

import pytest
from PIL import Image

from rag_ingest.batch import run_batch
from rag_ingest.engines import create_engine
from rag_ingest.engines.vision import HINT_INTRO, HintProvider, VisionEngine, _hint_provider
from rag_ingest.models import OcrLine, Recognition
from rag_ingest.stores import LineStore, VisionStore
from rag_ingest.textproc import NO_TEXT_MARK


class FakeTransport:
    model = "fake-model"

    def __init__(self):
        self.calls = []

    def complete(self, prompt, image):
        self.calls.append((prompt, image))
        return "อ่านได้"

    def settings(self):
        return {"backend": "fake"}


def write_lines(paths, folder, stem, lines):
    store = LineStore(paths.method(folder))
    store.folder.mkdir(exist_ok=True)
    store.path(stem).write_text("".join(f"{t}\tconfidence={c}\n" for t, c in lines),
                                encoding="utf-8")


def test_unknown_engine():
    with pytest.raises(ValueError, match="Unknown engine"):
        create_engine("nope")


def test_registry_does_not_import_heavy_libraries():
    code = ("import sys, rag_ingest.engines, rag_ingest.cli; "
            "print([m for m in ('easyocr', 'ollama', 'httpx', 'pytesseract') if m in sys.modules])")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_hints_single_source_unlabelled_both_labelled(paths):
    write_lines(paths, "easyocr", "img", [("easy line", 0.9), ("low", 0.3)])
    write_lines(paths, "tesseract", "img", [("tess line", 0.85)])

    assert _hint_provider(paths, "easyocr").hint("img") == ("easy line", ["easyocr"])
    assert _hint_provider(paths, "both").hint("img") == (
        "[easyocr OCR]\neasy line\n\n[tesseract OCR]\ntess line", ["easyocr", "tesseract"])
    assert HintProvider({}).hint("img") == (None, [])
    # Only one source has confident lines -> no label, like a single source.
    write_lines(paths, "tesseract", "img", [("tess line", 0.1)])
    assert _hint_provider(paths, "both").hint("img") == ("easy line", ["easyocr"])


def test_vision_engine_prompt_settings_and_upscale(paths, make_image):
    image = make_image("img.png", size=(100, 50))
    write_lines(paths, "easyocr", "img", [("hint", 0.95)])
    transport = FakeTransport()
    engine = VisionEngine(transport, "PROMPT", _hint_provider(paths, "easyocr"), upscale=200)

    result = engine.recognize(image)

    prompt, sent = transport.calls[0]
    assert prompt == "PROMPT" + HINT_INTRO + "hint"
    assert sent[:4] == b"\x89PNG"
    assert result.text == "อ่านได้" and result.model == "fake-model"
    assert result.settings == {"backend": "fake", "temperature": 0, "easyocr_hint": True,
                               "tesseract_hint": False, "upscale_min_side": 200}


def test_vision_engine_without_hint_sends_original_bytes(paths, make_image):
    image = make_image("img.png", size=(1200, 1100))
    transport = FakeTransport()
    VisionEngine(transport, "PROMPT", None, upscale=1000).recognize(image)
    prompt, sent = transport.calls[0]
    assert prompt == "PROMPT"
    assert sent == image.read_bytes()  # already big enough: not re-encoded


class FlakyEngine:
    store_kind = "lines"
    default_output = "x"

    def recognize(self, image):
        if "bad" in image.name:
            raise RuntimeError("boom")
        return Recognition("t", [OcrLine("t", 1.0)])


def test_batch_isolates_failures_and_skips_existing(paths, make_image, capsys):
    images = [make_image(n) for n in ("a.png", "bad.png", "c.png")]
    store = LineStore(paths.method("x"))

    first = run_batch(FlakyEngine(), store, images)
    assert (first.done, first.skipped, first.failed) == (2, 0, 1)
    assert "[2/3] bad.png - FAILED: boom" in capsys.readouterr().out

    again = run_batch(FlakyEngine(), store, images, skip_existing=True)
    assert (again.done, again.skipped, again.failed) == (0, 2, 1)


def test_vision_store_accepts_engine_output(paths, make_image):
    image = make_image("img.png")
    result = VisionEngine(FakeTransport(), "P").recognize(image)
    VisionStore(paths.method("vision"), paths).save(image, result)
    assert VisionStore(paths.method("vision"), paths).texts() == {"img": "อ่านได้"}


def test_tesseract_engine_reads_text(make_image):
    pytest.importorskip("pytesseract")
    try:
        engine = create_engine("tesseract", lang="eng")
        engine._tesseract.get_tesseract_version()
    except Exception:
        pytest.skip("tesseract binary not available")
    from PIL import ImageDraw, ImageFont
    img = Image.new("RGB", (600, 120), "white")
    ImageDraw.Draw(img).text((20, 30), "HELLO WORLD", fill="black",
                             font=ImageFont.load_default(size=48))
    path = make_image("t.png")
    img.save(path)
    result = engine.recognize(path)
    assert "HELLO" in result.text and result.lines and 0 <= result.lines[0].confidence <= 1


def test_formats_vision_models_may_reject_are_sent_as_png(paths, make_image):
    from rag_ingest.imaging import image_bytes
    from rag_ingest.readers.builder import pixel_size

    gif = paths.images / "shot.gif"
    Image.new("RGB", (300, 200), "white").save(gif)
    sent = image_bytes(gif)
    assert sent[:4] == b"\x89PNG" and pixel_size(sent) == (300, 200)
    png = make_image("shot.png", size=(300, 200))
    assert image_bytes(png) == png.read_bytes()  # PNG/JPEG go as-is
    assert pixel_size(b"not an image") == (0, 0)  # e.g. EMF: reported, not crashed


def test_results_record_the_prompt_actually_used(paths, make_image):
    from rag_ingest.engines.vision import _vision_engine

    image = make_image("img.png")
    store = VisionStore(paths.method("vision"), paths)
    custom = paths.root / "prompts" / "compact.txt"
    custom.write_text("SHORT PROMPT", encoding="utf-8")

    transport = FakeTransport()
    store.save(image, _vision_engine(transport, paths, None, 0, custom).recognize(image))
    assert transport.calls[0][0] == "SHORT PROMPT"
    assert store.load("img")["prompt_file"] == "prompts/compact.txt"

    store.save(image, _vision_engine(FakeTransport(), paths, None, 0, None).recognize(image))
    assert store.load("img")["prompt_file"] == "prompts/thai_document_ocr.txt"  # default unchanged

    with pytest.raises(FileNotFoundError, match="Prompt not found"):
        _vision_engine(FakeTransport(), paths, None, 0, paths.root / "missing.txt")


def lined_image(path, rows=(100, 300, 500), size=(200, 600)):
    """White image with black text-like bars 20 px high starting at the given rows."""
    img = Image.new("RGB", size, "white")
    for top in rows:
        img.paste((0, 0, 0), (10, top, 190, top + 20))
    img.save(path)
    return path


def test_cut_rows_land_between_lines(tmp_path):
    from rag_ingest.imaging import cut_rows, load_rgb

    img = load_rgb(lined_image(tmp_path / "tall.png"))
    cuts = cut_rows(img, 200)
    assert cuts and all(not (top <= cut < top + 20) for cut in cuts for top in (100, 300, 500))


def test_short_image_is_one_strip(tmp_path):
    from rag_ingest.imaging import strips, load_rgb

    assert len(strips(load_rgb(lined_image(tmp_path / "s.png", size=(200, 240))), 200)) == 1


def test_tiled_engine_reads_each_strip_and_joins(tmp_path):
    transport = FakeTransport()
    result = VisionEngine(transport, "PROMPT", tile_height=200).recognize(lined_image(tmp_path / "t.png"))
    assert result.text == "\n".join(["อ่านได้"] * len(transport.calls))
    assert result.settings["tiles"] == len(transport.calls) > 1


class BlankStripTransport(FakeTransport):
    def complete(self, prompt, image):
        self.calls.append((prompt, image))
        return NO_TEXT_MARK if len(self.calls) == 1 else "ข้อความ"


def test_strips_without_text_are_left_out(tmp_path):
    result = VisionEngine(BlankStripTransport(), "P", tile_height=200).recognize(lined_image(tmp_path / "b.png"))
    assert NO_TEXT_MARK not in result.text


def test_tiling_refuses_hints(paths):
    with pytest.raises(ValueError):
        VisionEngine(FakeTransport(), "P", _hint_provider(paths, "easyocr"), tile_height=200)
