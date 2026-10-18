# rag-ingestion-poc

Extract accurate text from Thai PDFs with embedded screenshots, for RAG ingestion.
The PDFs' own text layer is used directly; only the screenshots are OCR'd, and
PaddleOCR, Tesseract, a local vision model (Ollama `qwen2.5vl:3b`) and the
Typhoon OCR API are compared for that.

Full status, results and plan: [note.md](note.md).

## Setup
```
uv pip install --python .venv/bin/python -e ".[tesseract,api,dev]"   # Windows: .venv/Scripts/python.exe
```
Extras: `tesseract`, `paddle` (~1 GB), `ollama`, `api`, `dev`; install only the engines you use.
Tesseract needs the binary with Thai data (`brew install tesseract tesseract-lang`, or winget on Windows).
For the local vision model: `ollama pull qwen2.5vl:3b`.
Typhoon / other OpenAI-compatible API: set `OCR_API_URL`, `OCR_API_MODEL` (and `OCR_API_KEY` if needed).

## Pipeline
```
rag-ingest inspect                          # text chars + images per PDF page
rag-ingest extract-images                   # data/raw/*.pdf -> outputs/images/
rag-ingest ocr --engine api                 # or tesseract | paddle | ollama  (--hint, --upscale, --skip-existing)
rag-ingest review --dir outputs/vision_hint # correct results -> ground truth
rag-ingest benchmark                        # CER per method -> outputs/benchmark_report.md
rag-ingest compare --a outputs/vision --b outputs/vision_hint --diff
rag-ingest ingest                           # final RAG text -> outputs/ingest/<pdf>.md
```
`python -m rag_ingest ...` works the same. Every command has `--help`.

## Layout
```
src/rag_ingest/
  cli.py          subcommands; thin, only wires arguments to the modules below
  config.py       folder layout (Paths) and the known result folders (METHODS)
  models.py       ImageRef (the <pdf>_page_N_image_M naming), OcrLine, Recognition
  pdf.py          text blocks + embedded images with positions (PyMuPDF)
  stores.py       repositories for the two result formats (vision JSON, line .txt)
  engines/        OCR engines behind one interface, created by name, heavy imports deferred
    vision.py     prompt/hints/image prep, with Ollama or OpenAI-compatible transport
  batch.py        run an engine over a folder (resume, per-image failure isolation)
  quality.py      rules that reject bad OCR (empty, "no text", prompt echo)
  ingest.py       text layer + screenshot OCR merged in reading order
  compare.py, benchmark.py, metrics.py, textproc.py
  review/         local correction UI
tests/            pytest; no model or network needed
```

## Tests
```
.venv/bin/python -m pytest
```
