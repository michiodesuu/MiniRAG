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
rag-ingest review --dir outputs/vision_hint # correct a model's results
rag-ingest review --blank                   # type ground truth from the image alone -> outputs/ground_truth/
rag-ingest benchmark                        # CER + failures per method -> outputs/reports/benchmark_report.md
rag-ingest compare --a outputs/vision --b outputs/vision_hint --diff
rag-ingest ingest                           # final RAG text -> outputs/ingest/<pdf>.md
rag-ingest chunk                            # -> outputs/chunks/chunks.jsonl (token budget, guards)
rag-ingest embed                            # -> outputs/index/<run>/ vectors + manifest
rag-ingest search "ถ้ารายการถูกอนุมัติแล้วจะแก้ไขได้อย่างไร" --role branch_manager
rag-ingest eval-retrieval [--grid]          # -> outputs/reports/retrieval_report.md + .csv
```
### OCR prompts
| file | tokens* | use |
|---|---|---|
| `prompts/thai_document_ocr.txt` | 212 | default; all existing results were made with it |
| `prompts/thai_document_ocr_compact.txt` | 536 | stricter Thai/number/layout rules, explains the REFERENCE hints |
| `prompts/thai_document_ocr_full.txt` | 9,147 | full specification; only for models with a large context (too long for `qwen2.5vl:3b`, num_ctx 8192) |

\*Qwen2.5-VL tokenizer. Pick one with `rag-ingest ocr --engine ollama --prompt prompts/thai_document_ocr_compact.txt --output-dir outputs/vision_compact`; each result records its `prompt_file`. Compare against the default with `rag-ingest benchmark` (every `outputs/vision_*` folder is picked up) before switching.

### OCR engine
Ingest takes Typhoon OCR (`outputs/vision_typhoon`) and falls back to the local `vision_hint` result when Typhoon's text fails the quality rules. On 18 screenshots this scores mean CER 0.164, against 0.225 / 0.228 for either alone. See `note.md` 2.18. Ground truth lives in `outputs/ground_truth/`, and `provenance.json` records who made each file and how (`human` or `claude`; blank box or edited from a model). The benchmark reports each provenance separately.

`python -m rag_ingest ...` works the same. Every command has `--help`; `--docs-dir`
(alias `--pdf-dir`) points at another folder of documents.

### Chunks, vectors and search
Install the `embed` extra (`uv pip install -e ".[embed]"`, pulls in PyTorch). `chunk`
counts tokens with the embedding model's own tokenizer (`--embedder local --model …`;
`--embedder fake` for a quick run), keeps screenshots with the instruction before them,
never crosses a section, drops contents pages, and refuses chunks holding a Thai
national ID (exit code 1; positions only in `outputs/chunks/rejected.jsonl`).
Documents in `config.EXCLUDED_DOCUMENTS` are never read. `embed` caches vectors in
`outputs/index/cache/` (not committed); `search` and `eval-retrieval` refuse an index
whose source files changed. Models, licences and the exact text-to-vector method:
note.md 2.17. The embedding API backend reads `EMBED_API_URL`, `EMBED_API_MODEL`,
`EMBED_API_KEY`.

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
  quality.py      rules that reject bad OCR (empty, "no text", prompt echo, repetition loop)
  textproc.py     text normalization shared by ingest and scoring
  ingest.py       document text + screenshot OCR merged in reading order
  thai_glyphs.py  restores Thai vowels a PDF's fonts mislabel (shared by reader and converter)
  ocr_eval/       how good is the OCR?
    ground_truth.py   outputs/ground_truth/ + provenance (who made it, how)
    benchmark.py      CER and failures per method, the ingest "pipeline" row
    compare.py        two result folders side by side
    metrics.py        edit distance, CER
  review/         local correction UI (`review`, `review --blank`)
  convert/        PDF -> DOCX: preflight, layout (pdf2docx), docx_fixes, verify, report, cli
  chunking/       ingest .md -> chunks: units, sections, tables, text, builder, guards, pipeline
  embeddings/     fake / local / api embedders, model profiles, vector cache
  retrieval/      search and how good it is
    vectorstore.py    brute-force cosine search behind a VectorStore protocol
    indexing.py       index runs, manifests, staleness check
    evaluation.py     Recall@k, MRR, role accuracy, no-answer threshold
    report.py         outputs/reports/retrieval_report.md + .csv
    index_commands.py, eval_commands.py   `embed`, `search`, `eval-retrieval`
tests/            pytest; no model or network needed
```

## Tests
```
.venv/bin/python -m pytest
```
