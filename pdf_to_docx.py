#!/usr/bin/env python3
"""Convert PDF files to Word (.docx), keeping layout and formatting. Thai-aware.

    python pdf_to_docx.py data/raw/pdf/report.pdf                 # -> data/raw/docx/report.docx
    python pdf_to_docx.py data/raw/pdf/ --pages 1-3,8 --overwrite  # every PDF in a folder
    python pdf_to_docx.py a.pdf --output out/a.docx --report out/a_report.md

Same as `rag-ingest convert`. The work is done by the rag_ingest.convert
package (see its docstring for each step); this file is only the entry point.
Needs: uv pip install -e ".[convert]"
"""
import sys

from rag_ingest.convert.cli import main

if __name__ == "__main__":
    sys.exit(main(prog="pdf_to_docx.py"))
