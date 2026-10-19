"""WordprocessingML names and small lookups shared by the Word reader modules."""

NAMESPACES = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "v": "urn:schemas-microsoft-com:vml",
    "mc": "http://schemas.openxmlformats.org/markup-compatibility/2006",
}


def q(name: str) -> str:
    """'w:p' -> '{namespace}p', the tag form lxml uses."""
    prefix, local = name.split(":")
    return f"{{{NAMESPACES[prefix]}}}{local}"


# Structure
P, TBL, TR, TC = q("w:p"), q("w:tbl"), q("w:tr"), q("w:tc")
SDT, SDT_CONTENT, CUSTOM_XML = q("w:sdt"), q("w:sdtContent"), q("w:customXml")
PPR, TCPR, TXBX = q("w:pPr"), q("w:tcPr"), q("w:txbxContent")
# Inline content
T, TAB, BR, CR = q("w:t"), q("w:tab"), q("w:br"), q("w:cr")
RENDERED_BREAK = q("w:lastRenderedPageBreak")
ALTERNATE, CHOICE, FALLBACK = q("mc:AlternateContent"), q("mc:Choice"), q("mc:Fallback")
BLIP, IMAGEDATA = q("a:blip"), q("v:imagedata")
# Run properties (used by the PDF -> DOCX converter)
R, RPR, RFONTS, LANG, SECTPR = q("w:r"), q("w:rPr"), q("w:rFonts"), q("w:lang"), q("w:sectPr")
# Attributes
VAL, TYPE = q("w:val"), q("w:type")
R_EMBED, R_ID = q("r:embed"), q("r:id")


def val(element, child: str) -> str | None:
    """w:val of the first `child` (e.g. "w:numFmt") under element, or None."""
    if element is None:
        return None
    found = element.find(q(child))
    return None if found is None else found.get(VAL)
