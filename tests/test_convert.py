"""PDF -> DOCX converter (rag_ingest.convert) and its two entry points."""
import subprocess
import sys

import pytest
from lxml import etree

pytest.importorskip("pdf2docx")
from rag_ingest.convert import ConvertSettings, convert_pdf  # noqa: E402
from rag_ingest.convert.cli import main  # noqa: E402
from rag_ingest.convert.docx_fixes import FixCounts, fix_run_properties  # noqa: E402
from rag_ingest.convert.options import parse_pages  # noqa: E402
from rag_ingest.convert.verify import docx_page_texts  # noqa: E402
from rag_ingest.readers.word import ooxml as x  # noqa: E402

W_NS = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'


def xml(text: str):
    return etree.fromstring(text.replace("W_NS", W_NS, 1))


def tags(element) -> list[str]:
    return [etree.QName(child).localname for child in element]


def test_parse_pages():
    assert parse_pages(None, 5) is None
    assert parse_pages("1-3, 5, 2", 5) == [0, 1, 2, 4]
    with pytest.raises(ValueError, match="outside 1-5"):
        parse_pages("4-6", 5)


def test_run_gets_complex_script_twins_in_schema_order():
    run = xml("""<w:r W_NS><w:rPr><w:rFonts w:ascii="TH SarabunPSK" w:hAnsi="TH SarabunPSK"/>
                 <w:b/><w:sz w:val="32"/></w:rPr><w:t>ทดสอบ</w:t></w:r>""")
    counts = FixCounts()
    fix_run_properties(run, counts)
    rpr = run.find(x.RPR)
    assert tags(rpr) == ["rFonts", "b", "bCs", "sz", "szCs", "lang"]
    assert rpr.find(x.RFONTS).get(x.q("w:cs")) == "TH SarabunPSK"
    assert rpr.find(x.q("w:szCs")).get(x.VAL) == "32"
    assert rpr.find(x.LANG).get(x.q("w:bidi")) == "th-TH"
    assert (counts.runs, counts.complex_script_runs, counts.thai_runs) == (1, 1, 1)


def test_latin_run_is_not_tagged_thai_and_existing_values_kept():
    run = xml("""<w:r W_NS><w:rPr><w:rFonts w:ascii="Arial" w:cs="Tahoma"/><w:szCs w:val="20"/>
                 <w:sz w:val="24"/></w:rPr><w:t>Upload</w:t></w:r>""")
    counts = FixCounts()
    fix_run_properties(run, counts)
    rpr = run.find(x.RPR)
    assert rpr.find(x.RFONTS).get(x.q("w:cs")) == "Tahoma"
    assert [e.get(x.VAL) for e in rpr.iter(x.q("w:szCs"))] == ["20"]
    assert rpr.find(x.LANG) is None and counts.complex_script_runs == 0


def test_docx_page_texts_split_at_section_ends():
    body = xml("""<w:body W_NS><w:p><w:r><w:t>หนึ่ง</w:t></w:r><w:pPr><w:sectPr/></w:pPr></w:p>
                  <w:p><w:r><w:t>สอง</w:t></w:r></w:p><w:p><w:pPr><w:sectPr/></w:pPr></w:p></w:body>""")
    assert docx_page_texts(body) == ["หนึ่ง", "สอง\n"]


@pytest.fixture
def sample_pdf(tmp_path):
    import pymupdf
    from PIL import Image

    pdf_path, png = tmp_path / "sample.pdf", tmp_path / "shot.png"
    Image.new("RGB", (400, 200), "lightblue").save(png)
    with pymupdf.open() as doc:
        for n in (1, 2):
            page = doc.new_page()
            page.insert_text((72, 72), f"Page {n}: Accident report manual", fontsize=14)
            page.insert_image(pymupdf.Rect(72, 100, 472, 300), filename=str(png))
        doc.save(pdf_path)
    return pdf_path


def test_convert_pdf_end_to_end(sample_pdf, tmp_path):
    out = tmp_path / "out" / "sample.docx"
    report = convert_pdf(sample_pdf, out)
    assert out.exists()
    assert "Pages: 2 of 2 converted; DOCX sections: 2" in report
    assert "Text match with the PDF: mean 100.0%" in report
    with pytest.raises(FileExistsError, match="--overwrite"):
        convert_pdf(sample_pdf, out)
    assert "Pages: 1 of 2" in convert_pdf(sample_pdf, out, ConvertSettings(pages="2", overwrite=True))


def test_cli_batch_and_report(sample_pdf, tmp_path, capsys):
    code = main([str(sample_pdf), "--output-dir", str(tmp_path / "docx"),
                 "--report", str(tmp_path / "report.md")])
    assert code == 0
    assert (tmp_path / "docx" / "sample.docx").exists()
    assert "Done: 1 converted, 0 failed." in capsys.readouterr().out
    assert (tmp_path / "report.md").read_text(encoding="utf-8").startswith("# sample.pdf")


@pytest.mark.parametrize("command", [
    [sys.executable, "pdf_to_docx.py", "--help"],
    [sys.executable, "-m", "rag_ingest", "convert", "--help"],
])
def test_entry_points_share_options(command):
    from rag_ingest.config import Paths

    out = subprocess.run(command, cwd=Paths.default().root, capture_output=True, text=True, check=True)
    for option in ("--pages", "--keep-text", "--output-dir", "--report"):
        assert option in out.stdout
