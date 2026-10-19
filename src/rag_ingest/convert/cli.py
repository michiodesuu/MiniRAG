"""Command-line options and the batch loop, shared by `rag-ingest convert`
and the root-level pdf_to_docx.py so both accept exactly the same options."""
import argparse
import logging
import sys
from pathlib import Path

from rag_ingest.config import ENCODING, Paths
from rag_ingest.convert import ConvertSettings, convert_pdf
from rag_ingest.convert.options import collect_pdfs, output_path

USAGE_EXAMPLES = """examples:
  %(prog)s data/raw/pdf/report.pdf                 -> data/raw/docx/report.docx
  %(prog)s data/raw/pdf/ --pages 1-3,8 --overwrite  every PDF in a folder
  %(prog)s a.pdf --output out/a.docx --report out/a_report.md"""


def add_arguments(parser: argparse.ArgumentParser, paths: Paths) -> None:
    parser.add_argument("inputs", nargs="+", type=Path, help="PDF files and/or folders of PDFs")
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--output", type=Path, help="Output .docx (one input PDF only)")
    target.add_argument("--output-dir", type=Path, default=paths.documents / "docx",
                        help="Folder for the .docx files (default: data/raw/docx)")
    parser.add_argument("--pages", help='Pages to convert, e.g. "1-3,8" (default: all)')
    parser.add_argument("--password", help="Password for encrypted PDFs")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing .docx files")
    parser.add_argument("--keep-text", action="store_true",
                        help="Keep the PDF's text exactly (skip the Thai vowel repair)")
    parser.add_argument("--report", type=Path, help="Also save the report as markdown")
    parser.add_argument("--verbose", action="store_true", help="Show pdf2docx progress")


def run(args: argparse.Namespace) -> int:
    """Convert every input PDF; one failure does not stop the batch. Returns the exit code."""
    logging.basicConfig(level=logging.INFO if args.verbose else logging.ERROR, format="%(message)s")
    pdfs = collect_pdfs(args.inputs)
    if args.output and len(pdfs) > 1:
        sys.exit("--output takes one PDF; use --output-dir for several")
    settings = ConvertSettings(args.pages, args.password, args.overwrite, not args.keep_text)

    reports, failed = [], 0
    for pdf in pdfs:
        print(f"Converting {pdf.name} ...", flush=True)
        try:
            reports.append(convert_pdf(pdf, output_path(pdf, args.output, args.output_dir), settings))
            print(reports[-1] + "\n", flush=True)
        except Exception as error:  # one bad file must not stop a batch
            failed += 1
            print(f"FAILED {pdf.name}: {error}\n", file=sys.stderr, flush=True)
    if args.report and reports:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text("\n\n".join(reports) + "\n", encoding=ENCODING)
    print(f"Done: {len(reports)} converted, {failed} failed.")
    return 1 if failed else 0


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    sys.stdout.reconfigure(encoding=ENCODING)  # Thai on the Windows console
    parser = argparse.ArgumentParser(
        prog=prog, description="Convert PDF files to Word (.docx), keeping layout and formatting.",
        epilog=USAGE_EXAMPLES, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(parser, Paths.default())
    return run(parser.parse_args(argv))
