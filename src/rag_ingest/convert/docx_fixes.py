"""What Word needs for Thai that pdf2docx leaves out.

Word draws Thai with a run's *complex-script* properties. pdf2docx sets only
the Latin ones, so Thai would fall back to Word's default Thai font, size and
weight. Every run gets:
  - w:cs font = its Latin font (e.g. TH SarabunPSK)
  - w:szCs = w:sz, w:bCs = w:b, w:iCs = w:i (inserted right after their twin,
    which keeps the schema order Word expects)
  - w:lang bidi="th-TH" when it contains Thai (line breaking, proofing)
and, unless disabled, the Thai vowels the PDF's fonts mislabel are restored
(rag_ingest.thai_glyphs), also when a split vowel spans two runs.
"""
from dataclasses import dataclass
from pathlib import Path
import re

from rag_ingest.readers.word import ooxml as x
from rag_ingest.thai_glyphs import FontKey, repair_thai_vowels

THAI = re.compile("[฀-๿]")
COMPLEX_SCRIPT_TWINS = (("w:b", "w:bCs"), ("w:i", "w:iCs"), ("w:sz", "w:szCs"))
_AFTER_LANG = ("w:eastAsianLayout", "w:specVanish", "w:oMath")  # schema: these follow w:lang
_FALSE = ("0", "false", "off")


@dataclass
class FixCounts:
    runs: int = 0
    complex_script_runs: int = 0
    thai_runs: int = 0
    am_restored: int = 0  # "ส ำ" / "ท า" -> "ำ"
    aa_restored: int = 0  # mislabelled "ำ" -> "า"


def _mirror(rpr, latin: str, complex_script: str) -> bool:
    """Copy w:b -> w:bCs (etc.) when the Latin property is set and its twin is not."""
    from lxml import etree

    source = rpr.find(x.q(latin))
    if source is None or rpr.find(x.q(complex_script)) is not None:
        return False
    twin = etree.Element(x.q(complex_script), dict(source.attrib))
    source.addnext(twin)
    return True


def _set_complex_script_font(rpr) -> bool:
    fonts = rpr.find(x.RFONTS)
    if fonts is None or fonts.get(x.q("w:cs")):
        return False
    latin = fonts.get(x.q("w:ascii")) or fonts.get(x.q("w:hAnsi"))
    if not latin:
        return False
    fonts.set(x.q("w:cs"), latin)
    return True


def _set_thai_language(rpr) -> None:
    from lxml import etree

    lang = rpr.find(x.LANG)
    if lang is None:
        lang = etree.Element(x.LANG)
        follower = next((f for t in _AFTER_LANG if (f := rpr.find(x.q(t))) is not None), None)
        if follower is not None:
            follower.addprevious(lang)
        else:
            rpr.append(lang)
    lang.set(x.q("w:bidi"), "th-TH")


def fix_run_properties(run, counts: FixCounts) -> None:
    from lxml import etree

    counts.runs += 1
    rpr = run.find(x.RPR)
    if rpr is None:
        rpr = etree.Element(x.RPR)
        run.insert(0, rpr)
    changed = _set_complex_script_font(rpr)
    for latin, twin in COMPLEX_SCRIPT_TWINS:
        changed = _mirror(rpr, latin, twin) or changed
    counts.complex_script_runs += changed
    if THAI.search("".join(t.text or "" for t in run.iter(x.T))):
        _set_thai_language(rpr)
        counts.thai_runs += 1


def run_font(run) -> FontKey | None:
    """The run's font as the PDF named it: family plus bold, for matching defective fonts."""
    rpr = run.find(x.RPR)
    fonts = None if rpr is None else rpr.find(x.RFONTS)
    if fonts is None:
        return None
    bold = rpr.find(x.q("w:b"))
    return FontKey(fonts.get(x.q("w:ascii")) or "",
                   bold is not None and bold.get(x.VAL) not in _FALSE)


def repair_paragraph_vowels(paragraph, defective: set[FontKey], counts: FixCounts) -> None:
    texts = [(t, run_font(run) in defective)
             for run in paragraph.iter(x.R) for t in run.iter(x.T) if t.text]
    new, am, aa = repair_thai_vowels([(t.text, bad) for t, bad in texts])
    for (t, _), text in zip(texts, new):
        t.text = text
    counts.am_restored += am
    counts.aa_restored += aa


def apply_thai_fixes(docx: Path, repair_vowels: bool, defective: set[FontKey]) -> FixCounts:
    from docx import Document

    document = Document(str(docx))
    body = document.element.body
    counts = FixCounts()
    for run in body.iter(x.R):
        fix_run_properties(run, counts)
    if repair_vowels:
        for paragraph in body.iter(x.P):
            repair_paragraph_vowels(paragraph, defective, counts)
    document.save(str(docx))
    return counts
