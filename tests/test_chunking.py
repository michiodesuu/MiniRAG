"""Chunking (spec 3): small hand-written Thai ingest outputs, fallback token counter."""
import json
from pathlib import Path

import pytest

from rag_ingest.chunking import ChunkSettings, CorpusConfig, chunk_corpus, write_run
from rag_ingest.chunking.guards import redact_names, scan, valid_thai_id
from rag_ingest.chunking.pipeline import exclusion_reason
from rag_ingest.chunking.sources import DocumentImages
from rag_ingest.tokens import approximate_tokens

MD5_A, MD5_B = "a" * 32, "b" * 32
SCREEN_A = ("หน้าจอหลัก ปุ่มบันทึก\n"
            "<table><tr><td>ลำดับ</td><td>ชื่อ</td></tr><tr><td>1</td><td>สมชาย</td></tr></table>")
SCREEN_B = "หน้าจอรายการ ปุ่มแก้ไข ปุ่มลบ"

PDF_MANUAL = f"""<!-- page 1 -->
สารบัญ

1. เริ่มต้นใช้งาน ............................ 2

2. แก้ไขข้อมูล ................................ 3

3. ลบข้อมูล .................................. 4

<!-- page 2 -->
1. เริ่มต้นใช้งาน

ระบบนี้ใช้บันทึกอุบัติเหตุของสาขา

1.กดเลือกเมนู บันทึกอุบัติเหตุ

[ภาพหน้าจอ]
{SCREEN_A}
[/ภาพหน้าจอ]

<!-- page 3 -->
2. แก้ไขข้อมูล

กดปุ่มแก้ไขในรายการที่ต้องการ

[ภาพหน้าจอ]
{SCREEN_B}
[/ภาพหน้าจอ]

<!-- page 4 -->
3. ลบข้อมูล

กดปุ่มลบในรายการที่ต้องการ

[ภาพหน้าจอ]
{SCREEN_B}
[/ภาพหน้าจอ]

[ภาพหน้าจอ: ยังไม่ได้ OCR (คู่มือสาขา_page_4_image_2)]
"""

WORD_MANUAL = """<!-- page 1 -->
# บทนำ

ย่อหน้าแรกของคู่มือเขต

## ขั้นตอนอนุมัติ

| ชื่อ | สถานะ |
|---|---|
| คำขอ ก | อนุมัติแล้ว |
"""


def manifest_entry(page, index, md5, status, source="vision_typhoon"):
    return {"image": f"คู่มือสาขา_page_{page}_image_{index}", "document": "คู่มือสาขา.pdf",
            "page": page, "size": [800, 400], "md5": md5, "status": status, "source": source}


@pytest.fixture
def corpus(tmp_path):
    ingest, documents = tmp_path / "ingest", tmp_path / "raw"
    ingest.mkdir()
    (documents / "docx").mkdir(parents=True)
    (documents / "docx" / "คู่มือเขต.docx").write_bytes(b"")
    (ingest / "คู่มือสาขา.md").write_text(PDF_MANUAL, encoding="utf-8")
    (ingest / "คู่มือเขต.md").write_text(WORD_MANUAL, encoding="utf-8")
    manifest = [manifest_entry(2, 1, MD5_A, "ocr"), manifest_entry(3, 1, MD5_B, "ocr"),
                manifest_entry(4, 1, MD5_B, "duplicate"), manifest_entry(4, 2, "c" * 32, "needs_ocr")]
    cache = {MD5_A: {"text": SCREEN_A, "source": "vision_typhoon", "image": "x"},
             MD5_B: {"text": SCREEN_B, "source": "vision_typhoon", "image": "y"}}
    (ingest / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    (ingest / "ocr_cache.json").write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    return CorpusConfig(ingest, documents, {}, frozenset(),
                        {"คู่มือสาขา": "branch_manager", "คู่มือเขต": "district_manager"})


def run(config, **settings):
    return chunk_corpus(config, ChunkSettings(**settings), approximate_tokens)


def chunks_of(result, document):
    return [c for c in result.chunks if c["document"] == document]


def test_contents_page_dropped_and_steps_are_not_headings(corpus):
    result = run(corpus)
    assert result.documents["คู่มือสาขา"].toc_pages_dropped == [1]
    assert not any("....." in c["display_text"] for c in result.chunks)
    sections = {c["section"] for c in chunks_of(result, "คู่มือสาขา")}
    assert sections == {"1. เริ่มต้นใช้งาน", "2. แก้ไขข้อมูล", "3. ลบข้อมูล"}  # "1.กดเลือกเมนู" is a step


def test_screenshot_stays_with_its_instruction_and_whole(corpus):
    result = run(corpus)
    with_screen = [c for c in result.chunks if "หน้าจอหลัก" in c["display_text"]]
    assert len(with_screen) == 1
    assert "1.กดเลือกเมนู บันทึกอุบัติเหตุ" in with_screen[0]["display_text"]
    assert "หน้าจอหลัก ปุ่มบันทึก\n| ลำดับ | ชื่อ |" in with_screen[0]["display_text"]
    assert with_screen[0]["screenshot_md5"] == [MD5_A]
    assert with_screen[0]["kind"] == "mixed"


def test_sections_and_documents_never_mix(corpus):
    result = run(corpus)
    for chunk in result.chunks:
        assert not ("ปุ่มแก้ไขในรายการ" in chunk["embed_text"] and "ปุ่มลบในรายการ" in chunk["embed_text"])
    assert len({c["chunk_id"] for c in result.chunks}) == len(result.chunks)


def test_context_header_and_page_range(corpus):
    edit = next(c for c in run(corpus).chunks if c["section"] == "2. แก้ไขข้อมูล")
    assert edit["embed_text"].startswith("คู่มือสาขา › 2. แก้ไขข้อมูล\n")
    assert (edit["page_start"], edit["page_end"], edit["role"]) == (3, 3, "branch_manager")
    plain = next(c for c in run(corpus, context_header=False).chunks if c["section"] == "2. แก้ไขข้อมูล")
    assert not plain["embed_text"].startswith("คู่มือสาขา")


def test_chunk_ids_are_stable(corpus):
    first, second = run(corpus), run(corpus)
    assert [c["chunk_id"] for c in first.chunks] == [c["chunk_id"] for c in second.chunks]
    assert first.chunks[0]["chunk_id"].startswith("คู่มือสาขา#p2#c")


def test_duplicate_screenshot_embedded_once_and_placeholder_counted(corpus):
    result = run(corpus)
    embedded = sum(c["embed_text"].count(SCREEN_B) for c in result.chunks)
    assert embedded == 1
    delete = next(c for c in result.chunks if c["section"] == "3. ลบข้อมูล")
    assert SCREEN_B in delete["display_text"]
    assert SCREEN_B not in delete["embed_text"]
    assert "กดปุ่มลบในรายการ" in delete["embed_text"]   # its own instruction is kept
    assert (delete["duplicate_screenshots"], delete["missing_ocr"]) == (1, 1)
    assert "ยังไม่ได้ OCR" not in delete["embed_text"]


def test_word_sections_and_tables(corpus):
    word = chunks_of(run(corpus, tables="flatten"), "คู่มือเขต")
    assert word[0]["doc_type"] == "docx"
    table = next(c for c in word if "อนุมัติแล้ว" in c["embed_text"])
    assert table["section"] == "บทนำ › ขั้นตอนอนุมัติ"
    assert "ชื่อ: คำขอ ก, สถานะ: อนุมัติแล้ว" in table["embed_text"]
    kept = next(c for c in chunks_of(run(corpus, tables="keep"), "คู่มือเขต") if "อนุมัติแล้ว" in c["embed_text"])
    assert "| คำขอ ก | อนุมัติแล้ว |" in kept["embed_text"]


def test_html_table_in_screenshot_flattened(corpus):
    chunk = next(c for c in run(corpus, screenshot_tables="flatten").chunks if "หน้าจอหลัก" in c["embed_text"])
    assert "ลำดับ: 1, ชื่อ: สมชาย" in chunk["embed_text"]
    assert "<table>" not in chunk["embed_text"]


def test_screenshot_display_is_markdown_not_html(corpus):
    chunk = next(c for c in run(corpus).chunks if "หน้าจอหลัก" in c["display_text"])
    assert "<table>" not in chunk["display_text"]


@pytest.fixture
def columns_chunk(corpus):
    return next(c for c in run(corpus, screenshot_tables="columns").chunks if "หน้าจอหลัก" in c["embed_text"])


def test_columns_keeps_the_header(columns_chunk):
    assert "| ลำดับ | ชื่อ |" in columns_chunk["embed_text"]


@pytest.mark.parametrize("field", ["embed_text", "display_text"])
def test_columns_drops_sample_rows(columns_chunk, field):
    assert "สมชาย" not in columns_chunk[field]


def test_screenshot_mode_leaves_document_tables(tmp_path):
    result = run(big_table_corpus(tmp_path, 3), screenshot_tables="columns")
    assert "รายการทดสอบลำดับที่ 2" in result.chunks[0]["embed_text"]


def big_table_corpus(tmp_path, rows: int) -> CorpusConfig:
    ingest = tmp_path / "big"
    ingest.mkdir()
    lines = ["| เลขที่ | รายการ |", "|---|---|"] + [f"| {i} | รายการทดสอบลำดับที่ {i} |" for i in range(rows)]
    text = "<!-- page 1 -->\n# ตาราง\n\n" + "\n".join(lines) + "\n"
    (ingest / "ตาราง.md").write_text(text, encoding="utf-8")
    return CorpusConfig(ingest, tmp_path / "none", {}, frozenset(), {})


def test_big_table_split_by_rows_with_header_repeated(tmp_path):
    result = run(big_table_corpus(tmp_path, 40), max_tokens=80, overlap=0)
    parts = [c for c in result.chunks if "รายการทดสอบ" in c["embed_text"]]
    assert len(parts) > 1
    for part in parts:
        assert "| เลขที่ | รายการ |\n|---|---|" in part["embed_text"]
        assert part["tokens"] <= 80
    rows = [line for p in parts for line in p["embed_text"].splitlines() if "รายการทดสอบ" in line]
    assert len(rows) == 40  # every row once, none cut


def test_budget_respected_and_overlap_is_whole_lines(tmp_path):
    ingest = tmp_path / "long"
    ingest.mkdir()
    lines = [f"บรรทัดที่ {i} อธิบายขั้นตอนการใช้งานระบบอย่างละเอียด" for i in range(30)]
    (ingest / "ยาว.md").write_text("<!-- page 1 -->\n" + "\n".join(lines) + "\n", encoding="utf-8")
    config = CorpusConfig(ingest, tmp_path / "none", {}, frozenset(), {})
    result = run(config, max_tokens=60, overlap=20)
    assert len(result.chunks) > 1
    assert all(c["tokens"] <= 60 for c in result.chunks)
    for before, after in zip(result.chunks, result.chunks[1:]):
        first_line = after["embed_text"].split("\n")[1]  # after the header
        assert first_line in before["embed_text"].split("\n")  # a whole earlier line


def test_valid_thai_id_rejected_and_never_written(corpus, tmp_path):
    base = "310170012345"
    good = base + str((11 - sum(int(d) * (13 - i) for i, d in enumerate(base)) % 11) % 10)
    bad = base + str((int(good[-1]) + 1) % 10)
    assert valid_thai_id(good)
    assert not valid_thai_id(bad)
    text = (corpus.ingest_dir / "คู่มือเขต.md").read_text(encoding="utf-8")
    (corpus.ingest_dir / "คู่มือเขต.md").write_text(
        text + f"\nเลขบัตร {good}\n\n## อื่น ๆ\n\nเลขอ้างอิง {bad}\n", encoding="utf-8")
    result = run(corpus)
    assert result.pii_found
    assert result.documents["คู่มือเขต"].pii_rejections == 1
    assert any(bad in c["embed_text"] for c in result.chunks)    # failed checksum: kept
    assert not any(good in c["embed_text"] for c in result.chunks)
    write_run(result, tmp_path / "out", {"counter": "approximate"})
    for name in ("rejected.jsonl", "manifest.json", "chunks.jsonl"):
        assert good not in (tmp_path / "out" / name).read_text(encoding="utf-8")
    rejected = [json.loads(line) for line in (tmp_path / "out" / "rejected.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rejected[0]["rule"] == "thai_national_id"
    assert set(rejected[0]) >= {"start", "length", "field"}


@pytest.mark.parametrize("text, expected", [
    ("00840 : นายสมศักดิ์ รักเรียน\nตำแหน่ง: ผู้จัดการสาขา", "00840 : [ชื่อ]\nตำแหน่ง: ผู้จัดการสาขา"),
    ("ผู้ประสบเหตุ น.ส. สมหญิง ใจดี ได้รับบาดเจ็บ", "ผู้ประสบเหตุ [ชื่อ] ได้รับบาดเจ็บ"),
    ("นางสาวมาลี", "[ชื่อ]"),
    ("นายแพทย์สมชาย รักษา", "[ชื่อ]"),
    ("นายจ้างต้องแจ้งภายใน 48 ชั่วโมง", "นายจ้างต้องแจ้งภายใน 48 ชั่วโมง"),
    ("นายทะเบียน นายกรัฐมนตรี นางพยาบาล นางสาว", "นายทะเบียน นายกรัฐมนตรี นางพยาบาล นางสาว"),
])
def test_names_redacted_ordinary_words_kept(text, expected):
    assert redact_names(text)[0] == expected


def test_name_never_reaches_chunks_or_reports(corpus, tmp_path):
    text = (corpus.ingest_dir / "คู่มือเขต.md").read_text(encoding="utf-8")
    (corpus.ingest_dir / "คู่มือเขต.md").write_text(text + "\nผู้ใช้ นายสมชาย ใจดี เข้าระบบ\n", encoding="utf-8")
    result = run(corpus)
    assert result.documents["คู่มือเขต"].name_redactions == 1
    write_run(result, tmp_path / "out", {"counter": "approximate"})
    for name in ("chunks.jsonl", "manifest.json"):
        assert "สมชาย" not in (tmp_path / "out" / name).read_text(encoding="utf-8")
    assert any("[ชื่อ]" in c["embed_text"] and "[ชื่อ]" in c["display_text"] for c in result.chunks)


def test_overlap_cut_inside_a_name_does_not_keep_the_surname(corpus):
    # The overlap is whole space-separated phrases of the last line, so it can start at the surname.
    words = " ".join(f"คำ{i}" for i in range(60))
    text = (corpus.ingest_dir / "คู่มือเขต.md").read_text(encoding="utf-8")
    (corpus.ingest_dir / "คู่มือเขต.md").write_text(
        text + f"\n\n{words} นายสมชาย ใจดี\n\nข้อความถัดไป\n", encoding="utf-8")
    for max_tokens in range(30, 90, 5):
        for overlap in range(1, 12):
            for chunk in run(corpus, max_tokens=max_tokens, overlap=overlap).chunks:
                assert "ใจดี" not in chunk["embed_text"] + chunk["display_text"]


@pytest.mark.parametrize("text, expected", [
    ("นาย จ้าง", "[ชื่อ]"),                                  # a space after the title: a name, not a compound
    ("นาย๑๒๓", "นาย๑๒๓"),                                    # Thai digits are not name characters
])
def test_name_rule_edges(text, expected):
    assert redact_names(text)[0] == expected


def test_excluded_documents_skipped_and_reported(corpus):
    config = CorpusConfig(corpus.ingest_dir, corpus.documents_dir, {"คู่มือสาขา.pdf": "ข้อมูลส่วนบุคคล"},
                          frozenset({"คู่มือเขต"}), corpus.roles)
    result = run(config)
    assert result.chunks == []
    assert {(e["document"], e["reason"]) for e in result.excluded} == {
        ("คู่มือสาขา", "ข้อมูลส่วนบุคคล"), ("คู่มือเขต", "excluded on the command line (--exclude)")}


@pytest.mark.parametrize("evidence, excluded", [
    ({"pdf"}, True),            # the manifest says it came from the excluded PDF
    ({"docx"}, False),          # proven to come from another file type
    (set(), True),              # no evidence: fail closed
    ({"pdf", "docx"}, True),    # mixed evidence: fail closed
])
def test_exclusion_is_by_file_and_fails_closed(evidence, excluded):
    reason = exclusion_reason("คู่มือ", DocumentImages(source_types=evidence), set(), {"คู่มือ.pdf": "unredacted"})
    assert (reason is not None) == excluded


def test_renamed_copy_excluded_by_image_fingerprint():
    images = DocumentImages(source_types={"pdf"}, md5s={"f" * 32, "a" * 32})
    reason = exclusion_reason("ชื่อใหม่", images, set(), {"คู่มือ.pdf": "unredacted"}, {"f" * 32: "คู่มือ.pdf"})
    assert reason == "contains images of excluded file คู่มือ.pdf"


def test_name_matching_ignores_case_and_invisible_characters():
    images = DocumentImages()
    assert exclusion_reason("Report\u200b ", images, set(), {"report.PDF": "x"}) == "x"


@pytest.mark.parametrize("text", ["{id}, 0812345678", "{id} 25", "25 {grouped}", "{grouped} 081-234-5678",
                                  "{fours}", "| {cells} |", "{spaced}"])
def test_id_next_to_other_numbers_is_found(text):
    good = thai_id()
    grouped = f"{good[0]}-{good[1:5]}-{good[5:10]}-{good[10:12]}-{good[12]}"
    fours = f"{good[:4]} {good[4:8]} {good[8:12]} {good[12]}"          # 4-4-4-1, not the printed layout
    cells = " | ".join([good[:3], good[3:7], good[7:10], good[10:]])     # split across Markdown cells
    spaced = " ".join(good)
    rules = [f.rule for f in scan({"t": text.format(id=good, grouped=grouped, fours=fours,
                                                    cells=cells, spaced=spaced)})]
    assert rules.count("thai_national_id") == 1


def thai_id() -> str:
    base = "310170012345"
    return base + str((11 - sum(int(d) * (13 - i) for i, d in enumerate(base)) % 11) % 10)


def test_id_cut_by_a_chunk_split_is_still_rejected(tmp_path):
    """A long unit is split by lines; the ID's two halves land in different chunks."""
    good = thai_id()
    lines = [f"บรรทัดที่ {i} อธิบายขั้นตอนการใช้งานระบบอย่างละเอียด" for i in range(12)]
    lines[5] += f" เลขบัตร {good[:6]}-"
    lines[6] = f"{good[6:]} " + lines[6]
    ingest = tmp_path / "split"
    ingest.mkdir()
    (ingest / "แยก.md").write_text("<!-- page 1 -->\n" + "\n".join(lines) + "\n", encoding="utf-8")
    result = run(CorpusConfig(ingest, tmp_path / "none", {}, frozenset(), {}), max_tokens=60, overlap=20)
    assert result.pii_found
    kept = "".join(c["embed_text"] + c["display_text"] for c in result.chunks)
    assert good[:6] not in kept
    assert good[6:] not in kept   # neither half survives
    write_run(result, tmp_path / "out", {})
    assert good[6:] not in (tmp_path / "out" / "rejected.jsonl").read_text(encoding="utf-8")


def test_id_split_across_table_cells_is_rejected(tmp_path):
    good = thai_id()
    cells = "".join(f"<td>{p}</td>" for p in (good[0], good[1:5], good[5:10], good[10:12], good[12]))
    ingest = tmp_path / "cells"
    ingest.mkdir()
    (ingest / "ตาราง.md").write_text(f"<!-- page 1 -->\n[ภาพหน้าจอ]\n<table><tr>{cells}</tr></table>\n[/ภาพหน้าจอ]\n",
                                    encoding="utf-8")
    result = run(CorpusConfig(ingest, tmp_path / "none", {}, frozenset(), {}))
    assert result.pii_found
    assert result.chunks == []


def test_outputs_written(corpus, tmp_path):
    result = run(corpus)
    path = write_run(result, tmp_path / "out", {"counter": "approximate"})
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(result.chunks)
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["settings"]["max_tokens"] == 400
    assert set(manifest["sources"]) == {"คู่มือสาขา", "คู่มือเขต"}
    assert manifest["documents"]["คู่มือสาขา"]["toc_pages_dropped"] == [1]


def test_real_ingest_output_if_present():
    folder = Path(__file__).resolve().parents[1] / "outputs" / "ingest"
    if not (folder / "ผู้จัดการสาขา.md").exists():
        pytest.skip("ingest output not present")
    config = CorpusConfig(folder, folder.parent.parent / "data" / "raw", {}, frozenset(), {})
    result = run(config)
    assert result.documents["ผู้จัดการสาขา"].toc_pages_dropped == [1]
    assert not result.pii_found
