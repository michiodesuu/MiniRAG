# rag-ingestion-poc

Extract accurate text from Thai PDF and Word (.docx) documents with embedded
screenshots, for RAG ingestion. The documents' own text is used directly
(PDF text layer, Word paragraphs/tables); only the screenshots are OCR'd, and
PaddleOCR, Tesseract, a local vision model (Ollama `qwen2.5vl:3b`) and the
Typhoon OCR API are compared for that.

Full status, results and plan: [note.md](note.md).

## Setup
```
uv pip install --python .venv/bin/python -e ".[tesseract,api,dev]"   # Windows: .venv/Scripts/python.exe
```
Extras: `docx` (Word input), `tesseract`, `paddle` (~1 GB), `ollama`, `api`, `dev`; install only what you use.
Tesseract needs the binary with Thai data (`brew install tesseract tesseract-lang`, or winget on Windows).
For the local vision model: `ollama pull qwen2.5vl:3b`.
Typhoon / other OpenAI-compatible API: set `OCR_API_URL`, `OCR_API_MODEL` (and `OCR_API_KEY` if needed).

## Source documents
```
data/raw/
  pdf/    .pdf
  docx/   .docx
  txt/    .txt   (no reader yet: listed as skipped)
  md/     .md    (no reader yet: listed as skipped)
```
All sub-folders are searched and the file extension picks the reader. File
names must be unique across folders (`pdf/a.pdf` and `docx/a.docx` clash,
because image results are named after the file).

## Converting PDF to Word
`pdf_to_docx.py` (repo root), or `rag-ingest convert` with the same options, rebuilds a PDF as an editable .docx with the same page
size, margins, fonts, tables and images, and fixes what Thai needs in Word
(complex-script font/size/bold, Thai language tag, the broken "ำ").
```
uv pip install --python .venv/bin/python -e ".[convert]"
python pdf_to_docx.py data/raw/pdf/report.pdf            # -> data/raw/docx/report.docx
python pdf_to_docx.py data/raw/pdf/ --pages 1-3 --overwrite --report out/report.md
```
It prints a check per file: pages, images, fonts, and how closely each page's text
matches the PDF. Scanned pages convert as pictures only. The PDF's fonts must be
installed where the .docx is opened. Overlapping images (e.g. blur patches on a
screenshot) are flattened into one picture, so what was hidden stays hidden.

## Pipeline
```
rag-ingest inspect                          # text chars + images per page
rag-ingest extract-images                   # data/raw/**/*.pdf|*.docx -> outputs/images/
rag-ingest ocr --engine api                 # or tesseract | paddle | ollama  (--hint, --upscale, --skip-existing)
rag-ingest review --dir outputs/vision_hint # correct results -> ground truth
rag-ingest benchmark                        # CER per method -> outputs/benchmark_report.md
rag-ingest compare --a outputs/vision --b outputs/vision_hint --diff
rag-ingest ingest                           # final RAG text -> outputs/ingest/<pdf>.md
```
### OCR prompts
| file | tokens* | use |
|---|---|---|
| `prompts/thai_document_ocr.txt` | 212 | default; all existing results were made with it |
| `prompts/thai_document_ocr_compact.txt` | 536 | stricter Thai/number/layout rules, explains the REFERENCE hints |
| `prompts/thai_document_ocr_full.txt` | 9,147 | full specification; only for models with a large context (too long for `qwen2.5vl:3b`, num_ctx 8192) |

\*Qwen2.5-VL tokenizer. Pick one with `rag-ingest ocr --engine ollama --prompt prompts/thai_document_ocr_compact.txt --output-dir outputs/vision_compact`; each result records its `prompt_file`. Compare against the default with `rag-ingest benchmark` (add the new folder to `METHODS` in `config.py`) on hand-corrected images before switching.

`python -m rag_ingest ...` works the same. Every command has `--help`; `--docs-dir`
(alias `--pdf-dir`) points at another folder of documents.

### Word documents
Put `.docx` files in `data/raw/docx/`; every command handles both formats. Headings become
`#` lines, numbered lists keep their numbers (`1.`, `2.1`, `ก)`), tables become
markdown, text boxes are kept, and screenshots are placed where they appear.
Page numbers are approximate (Word has no fixed pages): taken from the page
breaks Word saved, else from hard page breaks. Not read: headers/footers,
footnotes, comments, linked images, old `.doc` files (save as `.docx`).
EMF/WMF drawings are listed as `unsupported_format` in the ingest manifest.

## Layout
```
src/rag_ingest/
  cli.py          subcommands; thin, only wires arguments to the modules below
  config.py       folder layout (Paths) and the known result folders (METHODS)
  models.py       ImageRef (the <doc>_page_N_image_M naming), Page, TextBlock, EmbeddedImage, ...
  readers/        one reader per document type, chosen by extension
    pdf.py        text blocks + images with positions (PyMuPDF)
    builder.py    PageBuilder for flow documents (numbers pages, images, order)
    word/         .docx: inline content, styles, numbering, page breaks, tables
  stores.py       repositories for the two result formats (vision JSON, line .txt)
  engines/        OCR engines behind one interface, created by name, heavy imports deferred
    vision.py     prompt/hints/image prep, with Ollama or OpenAI-compatible transport
  batch.py        run an engine over a folder (resume, per-image failure isolation)
  quality.py      rules that reject bad OCR (empty, "no text", prompt echo)
  ingest.py       document text + screenshot OCR merged in reading order
  thai_glyphs.py  restores Thai vowels a PDF's fonts mislabel (shared by reader and converter)
  convert/        PDF -> DOCX: preflight, layout (pdf2docx), docx_fixes, verify, report, cli
  compare.py, benchmark.py, metrics.py, textproc.py
  review/         local correction UI
tests/            pytest; no model or network needed
```

## Tests
```
.venv/bin/python -m pytest
```
