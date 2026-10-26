# rag-ingestion-poc — Working Notes

Last updated: 2026-10-16 (Tesseract, both-hint run 27/27, Typhoon API 27/27, benchmark script)

Goal: extract text from Thai PDFs (embedded images) accurately enough for RAG ingestion, using only local tools.

> **2026-10-18: code refactored into the `rag_ingest` package** (see README). Sections below are history and name the old scripts; they map to:
> `run_vision_ocr.py` -> `rag-ingest ocr --engine ollama|api` (`--hint` = old `--hint`, `--hint both` = old `--hint --hint-source both`) · `tesseract_batch.py` -> `ocr --engine tesseract` · `ocr_batch.py`/`ocr_image.py` -> `ocr --engine paddle` (writes `outputs/ocr/*.txt`, the format hints and the benchmark read) · `review_vision.py` -> `review` · `compare_vision.py` -> `compare` · `benchmark.py` -> `benchmark` · `ingest_pdf.py` -> `ingest` · `check_pdf.py` -> `inspect` · `extract_image.py` -> `extract-images` · `extract_pdf.py` -> `extract-text`.

---

## 1. Tech stack

| Part | Tool | Notes |
|---|---|---|
| Language / env | Python 3.14, `.venv` created with **uv** | The venv has **no pip**. Install with `uv pip install --python .venv/Scripts/python.exe <pkg>` |
| PDF / image extraction | PyMuPDF | `src/extract_pdf.py`, `src/extract_image.py`, `src/check_pdf.py` |
| Classic OCR | PaddleOCR (Thai) | `src/ocr_image.py`, `src/ocr_batch.py`; output in `outputs/ocr/*.txt` as `text<TAB>confidence=0.xxxx` |
| Vision-language OCR | Ollama + `qwen2.5vl:3b` | Runs on **CPU**; the only installed model. A 7B model does not fit the hardware |
| Image handling | Pillow 12.3.0 | Used for upscaling |
| OS / shell | Windows 10, PowerShell | Always run with the venv Python (system Python lacks `ollama` -> `ModuleNotFoundError`) |
| OCR API | Typhoon OCR 1.5 2B (company internal API) | `--backend api`; see 2.7 |
| Classic OCR #2 | Tesseract 5.4 (`tha+eng`) via pytesseract | Installed with winget; Thai data added by hand; `src/tesseract_batch.py` |

`requirements.txt`: PyMuPDF, ollama, pillow, pytesseract.

Folders: `prompts/` (OCR prompt), `outputs/images/` (27 extracted images), `outputs/ocr/` (PaddleOCR results), `outputs/tesseract/` (Tesseract results), `outputs/vision/` (baseline vision results), `outputs/vision_hint/` (PaddleOCR hint + upscale), `outputs/vision_both/` (PaddleOCR + Tesseract hints), `outputs/vision_typhoon/` (Typhoon API).

---

## 2. What has been done

### 2.1 `src/run_vision_ocr.py` (vision OCR runner)
Sends the image plus `prompts/thai_document_ocr.txt` to `qwen2.5vl:3b`; saves one JSON per image (`image`, `model`, `prompt_file`, `settings`, `text`, `review_status: pending`).

Options:
- `--image PATH` — one image.
- `--all [DIR]` — batch over a folder (default `outputs/images/`). A failing image does not stop the run; prints a done/skipped/failed summary.
- `--skip-existing` — skip images that already have a result JSON (resume).
- `--hint` — adds OCR lines with confidence >= 0.8 to the prompt as a reference the model is told not to fully trust.
- `--hint-source paddle|tesseract|both` — which OCR output feeds `--hint` (default `paddle`, i.e. `outputs/ocr/`; `tesseract` = `outputs/tesseract/`; `both` adds two labelled blocks). Each result JSON records `paddle_hint` and `tesseract_hint`. Hint coverage over the 27 images (conf >= 0.8): paddle 19, tesseract 20, both 19 (+1 tesseract-only), none 7. **Not yet run end to end** with tesseract/both.
- `--upscale PX` — enlarge images whose shorter side is below PX (max 4x, LANCZOS).
- `--output-dir DIR` — save elsewhere so settings can be compared side by side.
- Temperature fixed at 0. UTF-8 stdout forced (fixes garbled Thai in the Windows console).

### 2.2 `src/review_vision.py` (manual correction UI)
Local web page (`127.0.0.1:8765`): image on the left, editable model text on the right. Ctrl+Enter saves `corrected_text` and sets `review_status: corrected` in the JSON. Purpose: produce ground-truth pairs for measuring accuracy and, later, fine-tuning.
Usage: `python src/review_vision.py --dir outputs/vision_hint`

### 2.3 `src/compare_vision.py` (A/B comparison)
Compares two result folders (default `outputs/vision` vs `outputs/vision_hint`): per-image similarity, text lengths, optional line diff (`--diff`), report saved to `outputs/reports/compare_report.md`. If `corrected_text` exists it also reports character error rate (CER) for both runs.

### 2.4 Runs
- **Baseline `outputs/vision`**: complete, 27/27 images, 0 failures. Settings: no hint, no upscale. (Started before temperature 0 was added, so it used Ollama's default temperature.)
- **Hint + upscale `outputs/vision_hint`** (`--hint --upscale 1000`, temp 0): **27/27 saved, 0 failures** (final rerun of the last 3 images completed).
  - Attempt 1 was stopped by Claude Code (system memory critically low) after 16 images.
  - Resume run: **11 images FAILED** with `request (~4200-4460 tokens) exceeds the available context size (4096 tokens)` (HTTP 400). Cause: upscaling + hint text exceed Ollama's default 4096 context for `qwen2.5vl:3b`.
  - **Fix applied**: `NUM_CTX = 8192` in `src/run_vision_ocr.py` (passed as `options.num_ctx`, recorded in each JSON's `settings`). Rerun succeeded for 4 of the failed images (`สาขา_page_11_image_1`, `page_3_image_2`, `page_4_image_2`, `page_6_image_1`); that rerun was then stopped again for low memory. (The 4 `ผู้จัดการเขต` page_4/5 images had succeeded in the resume run itself.)
  - Final rerun: `สาขา_page_7_image_1`, `page_8_image_1`, `page_9_image_1` completed OK with the 8192 context.
  - Note: results from before the fix used the default context (4096), the later ones 8192; the mix is unlikely to change transcription much but is not controlled.
- **Comparison** (`src/compare_vision.py --diff`, saved to `outputs/reports/compare_report.md`), latest on **all 27 images**: 8/27 identical (the short ones), mean A~B similarity 0.58.
  - B (hint + upscale + temp 0) looked better by reading the diffs:
    - removed a 3,298-char repetition loop on `ผู้จัดการเขต_page_3_image_1` (503 chars clean in B);
    - fewer invented words (A "ลงบัมมูล" vs B "ลบข้อมูล");
    - more complete text on `สาขา_page_4_image_1`, `page_4_image_3`, `page_4_image_2`;
    - `ผู้จัดการเขต_page_5_image_1`: A output "no readable text" (`[ไม่พบข้อความที่อ่านได้]`), B transcribed 8 lines;
    - baseline invented foreign-script junk on `สาขา_page_7_image_1` (e.g. 番号, "месяцев"); B has none, and on `page_8_image_1` / `page_9_image_1` baseline gave 196 chars vs 873 for B (full 7-row table with names/dates/status; B's two outputs are identical, likely because the two images are the same screen);
    - `ผู้จัดการเขต_page_4_image_1`: A 366 chars vs B 1,016 (B contains the full table rows with names/dates/status).
  - B possible regressions / unclear: `สาขา_page_5_image_1` shrank 1834 -> 848 chars and lost real content; "รายงานการขออนุมัติ" (B) vs "รายงานการอุบัติเหตุ" (A) on `ผู้จัดการเขต_page_2_image_1`; "ตกลง!" vs "คลิก!"; heading "ส่วน" dropped on `page_4_image_1`. Names/numbers still misread in both.
  - Caveat: judged by reading Thai in the diffs, **not verified against images**; no ground truth, so no CER. Baseline also ran without temperature 0, so the gain cannot be split between the hint, upscale and temperature.
- Sanity check on the first image: baseline transcription of `ผู้จัดการเขต_page_2_image_1.png` was correct.

### 2.6 Tesseract (added)
- **Install:** `winget install UB-Mannheim.TesseractOCR` (v5.4.0) -> `C:\Users\BD.V211\AppData\Local\Programs\Tesseract-OCR\tesseract.exe` (user-level install, no admin). The installer ships only `eng`/`osd`, so Thai data was downloaded by hand: `tha.traineddata` from `tesseract-ocr/tessdata_best` (7.6 MB) into that install's `tessdata/` folder. `tesseract --list-langs` now shows eng, osd, tha.
- **Python:** `pytesseract 0.3.13` installed with uv and added to `requirements.txt`.
- **`src/tesseract_batch.py`:** runs `lang="tha+eng"` (default `--psm 3`, option `--psm`, `--skip-existing`) over `outputs/images/`; groups words into lines, strips the spaces Tesseract inserts between Thai characters, averages word confidence per line, and writes `outputs/tesseract/<image>.txt` as `text<TAB>confidence=0.xxxx` (same format as the PaddleOCR files). Prints one line per image.
- **Run:** 27/27 images, 0 failures, a few seconds each (far faster than the vision model).
- **Results vs PaddleOCR vs vision model** (line counts; vision = hint run):
  - **Empty output on 7 images** (all tiny crops): `สาขา_page_2_image_1`, `3_image_1`, `3_image_3`, `7_image_2`, `8_image_2`, `เขต_page_4_image_2`, `4_image_3`. PaddleOCR is also empty / near-empty (1 line at conf 0.19-0.39) on 6 of them. The vision model reads 12-24 chars on every one of them, so only it handles these.
  - **Accurate where it reads:** e.g. `สาขา_page_4_image_3` is almost perfect Thai. Minor errors: lost the tone mark in "ซ้ำ" (printed "ซํา"/"ช้า"), added "๑" before bullets, "ตักเดือน" for "ตักเตือน". The vision model got that text right.
  - **Misses small text on big screenshots:** e.g. 14 lines vs PaddleOCR 82 on `เขต_page_3_image_1`; 17 vs 103 on `เขต_page_4_image_1`; 20 vs 70 on `สาขา_page_7/8/9_image_1`.
  - **Confidence is not comparable:** its scores (0.78-0.93) only rate the words it did read, not whether it found all the text.
  - **Not measured:** accuracy (no corrected text yet); judged by reading output.

### 2.7 Both-hint run and Typhoon API
- **`outputs/vision_both`** (`--hint --hint-source both --upscale 1000`, temp 0, ctx 8192, Ollama `qwen2.5vl:3b`): **27/27 saved, 0 failures.** Slow images took ~15 min (e.g. `สาขา_page_7_image_1`).
- **Typhoon OCR API** (company internal, OpenAI-compatible): `POST $OCR_API_URL` (set in `.env`, see `.env.example`), model `$OCR_API_MODEL` (`typhoon-ai/typhoon-ocr1.5-2b`). Reachable from this PC, no API key needed. Added `--backend api --api-url ... --api-model ...` to `run_vision_ocr.py` (key only via env `OCR_API_KEY`; URL/model also via `OCR_API_URL` / `OCR_API_MODEL`; internal URL intentionally not hardcoded). Batch of 27 finished in a few minutes -> `outputs/vision_typhoon/` (plain prompt, no hint, no upscale), 0 failures.
  - Looks fuller/more accurate on big screens (e.g. `เขต_page_3_image_1`: 1805 chars vs 503; correct place names) and on clean prose; still weak on tiny crops (`เขต_page_4_image_2` returned 1 char; `สาขา_page_8_image_2` gave `[ถังขยะรีไซเคิล]`).
  - Output contains HTML tables (`<table><tr><td>`) and markdown (`# `, `* `): decide whether to keep (structure) or strip to plain text before chunking. Comparison with hint run: `outputs/reports/compare_hint_vs_typhoon.md` (mean similarity 0.63).

### 2.8 `src/benchmark.py` (evaluation)
Scores every method against hand-corrected text: vision, vision_hint, vision_both, vision_typhoon, PaddleOCR (`outputs/ocr`), Tesseract. Ground truth = `corrected_text` saved by `review_vision.py` in any `outputs/vision*` folder, or `outputs/ground_truth/<image>.txt`. Normalizes (strips HTML/markdown, ignores whitespace), character error rate capped at 1.0 per image, mean + pooled CER, empty-output count, split into short (<=60 chars) and long references. Writes `outputs/reports/benchmark_report.md` and `outputs/reports/benchmark.csv`. Run: `python src/benchmark.py`.
- **First real result, only 4 corrected images** (`สาขา_page_10_image_1`, `11_image_1`, `2_image_1`, `8_image_3`; just 1 long): mean CER typhoon 0.075, vision 0.100, vision_hint 0.137, vision_both 0.153, PaddleOCR 0.433, Tesseract 0.529. **Not a verdict**: tiny sample; on the one long image the hint runs scored worse than baseline (0.55-0.61 vs 0.24); and the corrections were saved in `outputs/vision/` (edited from the baseline model's text), which can bias the ground truth toward the baseline.

### 2.9 Tesseract settings test + comparison hint column (2026-10-18, macOS)
- `src/tesseract_batch.py` now also takes `--upscale PX` and `--output-dir`, and uses the Windows exe path only if it exists (works with Homebrew Tesseract on macOS). pytesseract installed into `.venv`.
- Tested on the 7 empty crops (psm 3, 6, 6 + upscale 1000, 11 + upscale 1000): **no setting recovers real text**; at best 1-2 stray characters of noise (`|`, `of`, `x`). Overall output size barely changes (10.3k-11.1k chars over 27 images). Conclusion: tiny crops stay with the vision model; no change to the default `--psm 3`.
- `src/compare_vision.py` now shows which hints reached each prompt (columns `hint A` / `hint B`: P = PaddleOCR, T = Tesseract). Reads the flags from each JSON's `settings`.
- Observed in `vision_both`: `สาขา_page_7/8/9_image_1` are 4,462 chars each (vs 873 in `vision_hint`) and identical to each other. **Confirmed repetition loop**: 231 lines but only 6 unique (the same 3-4 menu lines repeated). That run is unusable for those 3 images; rerun with a repetition penalty / `num_predict` cap or exclude them from comparisons.

### 2.10 `src/ingest_pdf.py` — text layer + filtered OCR, merged in place (2026-10-18)
Model-free (low memory). Run: `python src/ingest_pdf.py` -> `outputs/ingest/<pdf>.md`, `manifest.json`, `needs_ocr.txt`, `ocr_cache.json`.
- **The PDFs have a clean Thai text layer** (66-1,199 chars per page), so the body text needs no OCR at all. Only screenshots are OCR'd.
- Repairs a text-layer defect: `ำ` extracted as space + `า` (`ท าการ` -> `ทำการ`) on 5 pages of `สาขา`. Safe because no Thai word starts with `า`.
- Skips icons (shorter side < 120 px, `--min-side`): the 7 "tiny crops" are icons (32x29 to 173x99). **Correction to 2.6:** the vision model's 24-char answer on them was `[ไม่พบข้อความที่อ่านได้]` (the no-text marker), not real text; Typhoon says the same. Nothing to read there; drop them from the benchmark.
- Dedups by md5: `สาขา_page_7/8/9_image_1` are the same file -> 18 unique images to OCR instead of 27.
- Takes OCR text from existing folders (`--ocr-dir`, repeatable; default `vision_typhoon` then `vision_hint`) and inserts it at the image's position on the page as `[ภาพหน้าจอ] ... [/ภาพหน้าจอ]`.
- **Typhoon failed on 2 of 18 unique screenshots**, now auto-rejected and replaced from `vision_hint`:
  - `สาขา_page_7_image_1` (also 8/9): returned its own built-in instructions ("Extract all text from the image ... Formatting Rules ...") instead of a transcription.
  - `สาขา_page_7_image_3`: returned `[ไม่พบข้อความที่อ่านได้]` on a 1400x777 form full of readable text.
  - Typhoon's benchmark lead (2.8) is on 4 images that don't include these; its failure rate needs to be counted when choosing the main OCR.

### 2.11 Refactor (2026-10-18)
- One package `src/rag_ingest/` + `pyproject.toml` (optional extras per engine, so a low-memory machine can skip PaddleOCR) + one CLI `rag-ingest`. Old `src/*.py` scripts removed.
- Structure: repositories own the two result formats (before, 5 scripts parsed them separately); OCR engines share one interface and are created by name with heavy imports deferred; the vision engine is split into prompt/hint/image prep and a swappable transport (Ollama / OpenAI-compatible API); one batch runner (resume, failure isolation) instead of three copies; OCR quality checks are named rules; ingestion is cache -> source chain -> page assembly.
- **Verified unchanged** against outputs captured before the refactor: `compare` (2 runs), `benchmark` report + CSV, `inspect`, `extract-images` (27 files), `extract-text`, Tesseract `ocr` (27 files), `ingest` markdown + cache + needs_ocr: all byte-identical. Vision engines could not be run here (no Typhoon access, model not pulled); their prompt, hint text, image bytes and hint flags were checked equal to the old code on 162 cases with a fake transport.
- Intended changes: `ingest` manifest `rejected` now records the reason (`{"source", "reason"}`); new results store image paths with `/` (old `\` paths still read correctly, which also fixes the review UI on macOS); review UI no longer serves files outside the reviewed folder via `/image/../x`; PaddleOCR now writes per-image `.txt` (the old batch wrote an unused `ocr_results.jsonl`).
- Tests: `python -m pytest` (36 tests, about 2 s, no model or network).

### 2.12 Word (.docx) input (2026-10-19, branch `feature/docx-ingestion`)
- Every command now reads `.pdf` and `.docx` from `data/raw/` (`--docs-dir`, alias `--pdf-dir`). Readers are chosen by extension (`readers/__init__.py`); everything after reading is unchanged.
- Word reader (`readers/word/`, python-docx + lxml, extra `docx`): headings -> `#`; list numbers recomputed from numbering.xml (decimal, letters, roman, Thai digits/letters, start overrides, nested `%1.%2`); tables -> markdown with merged cells aligned and `|` escaped, screenshots in cells placed after the table; text boxes kept after their paragraph; drawings stored twice (modern + fallback) read once; tracked deletions and field codes dropped; content controls (e.g. TOC) unwrapped.
- Page numbers: from Word's saved layout breaks (`w:lastRenderedPageBreak`) if present, else hard page breaks / "page break before" / next-page section breaks. Approximate either way.
- Images: GIF/BMP/TIFF accepted by the OCR batch; formats vision models may reject are sent as PNG (also fixes WebP being labelled JPEG for the API). EMF/WMF cannot be decoded -> manifest status `unsupported_format`.
- Manifest: new `document` key (file name) per image; the old `pdf` key is gone. `source` still means the OCR folder used.
- Structure: Strategy + registry (readers), Builder (`PageBuilder`, directed by `WordReader`), dispatch table (XML walker), Strategy (page-break policy), memoized style lookups, table parsing separated from markdown rendering. Max cognitive complexity in the package: 9.
- Verified: all PDF outputs byte-identical to the pre-refactor baseline (manifest differs only by `pdf` -> `document`). Word reader tested on generated Thai documents (64 tests in total); **not yet run on a real company .docx**.

### 2.13 `data/raw/` split by type (2026-10-19)
- Layout: `data/raw/pdf/`, `docx/`, `txt/`, `md/` (empty folders kept with `.gitkeep`). All PDFs moved into `pdf/`: the 2 original manuals (tracked, moved with `git mv`) and 3 new, untracked PDFs added by the user.
- `list_documents` now searches sub-folders; `.gitkeep`, `.DS_Store`, hidden files and Word lock files are ignored; files with no reader (`.txt`, `.md` for now) are reported on stderr and skipped; names must stay unique across folders.
- The 2 original manuals give byte-identical ingest output from the new location. `outputs/ingest/` has not been regenerated with the 3 new PDFs.

### 2.14 PDF -> DOCX converter (2026-10-19)
- `pdf_to_docx.py` at the repo root (pdf2docx + Thai post-processing + verification report; extra `convert`). 78 tests.
- Converted `data/raw/pdf/1.คู่มือเจ้าหน้าที่วินิจฉัยเงินทดแทน.pdf` -> `data/raw/docx/` (PDF untouched): 20/20 pages, page 13 is a scan (picture only), font TH SarabunPSK, 108 runs given Thai complex-script font/size/bold. 58 PDF image placements -> 36 DOCX pictures: 122x122 blur patches stacked on screenshots are flattened into one picture each.
- **The PDF's text layer is wrong for Thai vowels**, so the first DOCX had ~350 wrong words ("เจ้ำหน้ำที่", "สถำนพยำบำล", "ส ำนัก"). Cause, from the glyphs: Word draws "ำ" as a circle glyph (286) + the "า" glyph (263); the PDF labels the circle as a space and, in the regular font only, glyph 263 as "ำ" (the bold font labels it "า"). Only glyph 263 differs between the two fonts; no other glyph is mislabelled. Truth: circle + 263 = "ำ" (44), 263 alone = "า" (381).
- The converter now detects this from any PDF (a space-labelled glyph drawn right before an "ำ"-labelled glyph, which real Thai never has) and restores 44 "ำ" and 307 "า".
- Checked every word of the DOCX: (1) text equals the PDF's text except page-number position; (2) no invalid Thai sequence in 4,312 Thai characters; (3) 782 Thai words, 2 not in the PyThaiNLP dictionary, both correct (`กท.16`/`กท.44`, `ฯ`); (4) Tesseract OCR of the rendered pages (independent of the text layer): every disagreement reviewed: OCR dropping tone marks, "ํา" for "ำ" (confirms each restored "ำ"), garbled English, or lines next to screenshots, all checked by eye against the rendered page.
- **Privacy: the PDF's blur is fake redaction.** The full screenshots under the blur patches are still in the PDF, with readable full names and 13-digit national ID numbers (27 screenshots). `extract-images` / `ingest` on the PDF would OCR them into RAG text. The DOCX contains none of those originals (md5-checked).
- `rag_ingest`'s PDF reader has the same vowel problem for this PDF (it only repairs the "ท าการ" form); not fixed yet.
- The PDF and DOCX share a name, so `rag-ingest` stops ("Documents share a name") until one is moved, renamed or removed.

### 2.15 Shared Thai vowel repair; converter moved into the package (2026-10-19)
- `src/rag_ingest/thai_glyphs.py` is the one Thai vowel repair, used by the PDF reader (ingest) and the converter. The old partial repair (`textproc.fix_sara_am`, "ท าการ" only) is removed. A split "ำ" now also needs the circle to follow a consonant (with its marks), so stray marks are never joined.
- Effect: `ingest` of `1.คู่มือเจ้าหน้าที่วินิจฉัยเงินทดแทน.pdf` now gives the correct words (44 "ำ", 381 "า"; no "เจ้ำ"/"สถำน"/"ส ำ"). The 2 original manuals give byte-identical ingest output.
- Converter logic moved from the root script into `src/rag_ingest/convert/` (preflight, layout, docx_fixes, verify, report, cli); `pdf_to_docx.py` is now a short wrapper, and `rag-ingest convert` takes the same options. Both produce a DOCX identical to the current one (all 52 parts).
- 79 tests; max cognitive complexity 9; no repeated literals.

### 2.16 Prompt variants and `--prompt` (2026-10-19)
- Proposed long prompt saved as `prompts/thai_document_ocr_full.txt` (9,147 tokens, Qwen2.5-VL tokenizer). It exceeds `qwen2.5vl:3b`'s num_ctx 8192 on its own (plus image tokens and hints), so it cannot run locally as-is; Typhoon OCR is trained around its own fixed prompt, so a long custom prompt is not expected to help there either.
- `prompts/thai_document_ocr_compact.txt` (536 tokens): its key rules plus how to treat the REFERENCE hint block. The default prompt (212 tokens) is unchanged.
- `rag-ingest ocr --prompt FILE` (vision engines); each result's `prompt_file` now records the prompt actually used (before, it always named the default file).
- **Not yet tested:** compare compact vs default with `rag-ingest benchmark` on the corrected images (needs Ollama + the model, or the Typhoon API, on the Windows machine).

### 2.17 Chunking, embeddings and retrieval evaluation (2026-10-20, branch `feature/chunk-and-embed`)

New stages: `rag-ingest chunk | embed | search | eval-retrieval`. Input is the existing `outputs/ingest/` (the 2 original manuals; not regenerated). The unredacted PDF moved to `data/excluded/` (user's decision) and is on `config.EXCLUDED_DOCUMENTS`.

#### How text becomes a vector (exactly)

**1. What text is embedded: `embed_text`.** One chunk of one section of one document:
- the context header `<document> › <section>` on the first line (e.g. `ผู้จัดการสาขา › 6. ส่งคำขอแก้ไขและลบ`), so near-identical steps in the branch and district manuals stay distinguishable;
- then the chunk's units in reading order: text-layer blocks, headings, tables (Markdown, or `header: value` lines with `--tables flatten`), and screenshot OCR without the `[ภาพหน้าจอ]` tags;
- Unicode NFC; no page comments; placeholders (screenshots without OCR) and screenshots already embedded elsewhere contribute nothing;
- it may start with up to `--overlap` tokens of whole lines carried from the previous chunk of the same section.

Real example, `ผู้จัดการสาขา#p9#c13` (174 tokens):
```
ผู้จัดการสาขา › 6. ส่งคำขอแก้ไขและลบ
6. ส่งคำขอแก้ไขและลบ
หมายเลข 1 : กดเลือกที่เมนู ติดตามสถานะ ตามในรูปภาพ
...
ส่งคำขอแก้ไข
ต้องการส่งคำขอแก้ไขข้อมูลใช่หรือไม่          <- screenshot OCR (the pop-up)
ตกลง ยกเลิก
จากนั้นกดตกลงเพื่อทำการยืนยันการขแก้ไขข้อมูล *เมื่อขอแก้ไขสำเร็จต้องรอ ...
```
`display_text` keeps the original Markdown of the same span (with screenshot tags) for people and citations.

**2. Tokenisation.** e5-small, e5-base and bge-m3 all use the XLM-RoBERTa SentencePiece tokenizer (unigram model, 250,002 pieces). Its vocabulary holds whole Thai words, so Thai is split into words although Thai has no spaces. Real example, query `ถ้ารายการถูกอนุมัติแล้วจะแก้ไขได้อย่างไร`:
- bge-m3 (no prefix), 11 tokens: `<s> ▁ถ้า รายการ ถูก อนุมัติ แล้ว จะ แก้ไข ได้ อย่างไร </s>`
- e5-small (prefix `query: `), 14 tokens: `<s> ▁que ry : ▁ถ้า รายการ ถูก อนุมัติ แล้ว จะ แก้ไข ได้ อย่างไร </s>`

Chunk budgets (`--max-tokens`) are counted with this same tokenizer, header and prefix included. Past `max_seq_length` (512 for e5, set to 1024 for bge-m3) the encoder silently truncates the end; chunks are at most 600 tokens, so only e5 at `--max-tokens 600` can truncate.

**3. Encoder.** The transformer (12 layers for e5-small/-base, 24 for bge-m3) turns the token sequence into one contextual vector per token (384 / 768 / 1024 numbers).

**4. Pooling** (from each model's `1_Pooling/config.json`, not memory): e5 = mean of the token vectors (padding excluded); bge-m3 = the vector of the first token (`<s>`, "CLS"). e5 requires the prefixes `query: ` for questions and `passage: ` for chunks (its model card); bge-m3 needs none.

**5. L2 normalisation.** Each vector is divided by its length, so it has length 1 (the models' `Normalize` layer; checked again in code). For unit vectors a·b = |a||b|cos θ = cos θ, so the dot product is the cosine similarity.

**6. Search.** `vectors.npy` is an N × dim float32 matrix (N = number of chunks). A query vector q is scored against every row: scores = V·q (exact brute force; fast at N in the hundreds). Optional filters (document, role, doc_type) remove rows first; the top k scores are returned. Scores lie in [-1, 1]; in practice related Thai passages score high (0.8-0.9 for e5) and unrelated ones lower, but the scale differs per model, so the "not found" threshold is learned per configuration from the no-answer questions (`eval-retrieval`).

**7. Reproducibility.** Model revisions are pinned (commit hashes in `embeddings/profiles.py`) and recorded with every setting in each index manifest (`settings()`: model, revision, dim, pooling, max_len, prefixes, library versions). Vectors are cached under `outputs/index/cache/` with key sha256(settings JSON + NUL + embed_text), so a change of model, revision, setting or text gives a new key. Each index records the sha256 of every source `.md` and of `chunks.jsonl`; `search` and `eval-retrieval` refuse a stale index and name the changed file.

#### Models (from their model cards and repository files, 2026-10-19)

| model | licence | parameters | dim | max input | pooling | prefixes | revision |
|---|---|---|---|---|---|---|---|
| BAAI/bge-m3 | MIT | ~568M (2.27 GB fp32) | 1024 | 8192 | CLS | none ("no longer requires adding instructions") | 5617a9f |
| intfloat/multilingual-e5-base | MIT | 278M | 768 | 512 | mean | `query: ` / `passage: ` | d128750 |
| intfloat/multilingual-e5-small | MIT | 118M | 384 | 512 | mean | `query: ` / `passage: ` | 614241f |
| airesearch/WangchanX-Legal-ThaiCCL-Retriever | MIT | 568M | 1024 | 8192 | CLS | none assumed (card gives no usage; base bge-m3) | 2a03371 |
| kornwtp/simcse-model-m-bert-thai-cased | Apache-2.0 | ~178M (711 MB fp32) | 768 | 512 | CLS | none | 10ca1e5 |

Left out: kornwtp/ConGen-simcse-model-roberta-base-thai (card says Apache-2.0 but does not name its base model, so the licence chain cannot be checked). All candidates run locally; no external embedding API is used.

#### Decisions and deviations from the spec
- **Repeated screenshots:** the spec drops a chunk whose screenshot set equals an earlier chunk's. In the branch manual pages 7, 8, 9 show one screenshot under *different* instructions (edit, delete, request a change); dropping those chunks would delete the delete/request instructions from search. Instead the repeated screenshot's text is embedded once (counted as `duplicate_screenshots`) and each page keeps its own instructions. Whole chunks are still dropped when their `embed_text` is identical, or when they are screenshot-only with a repeated set.
- **Personal data:** a Thai national ID (13 digits, checksum verified) rejects the chunk and stops the run (exit 1) unless `--allow-pii`; rejected chunks never enter chunks.jsonl either way, and reports hold rule + position only. Each whole document is scanned before chunking (so a chunk split cannot cut an ID in two), and every chunk again. Digits are matched across separators, HTML/Markdown cells, Thai and full-width digits. Measured false IDs: 0 in 2,000 table rows of small numbers or of dates/amounts; ~8% of rows in tables dense with 1-7-digit numbers (privacy first). Known gap: an ID in an unusual grouping of 4+ parts directly next to another number. Phone numbers are flagged, not rejected: 2 in a form screenshot of the branch manual (`เบอร์โทรศัพท์`).
- **Exclusion** is by source file name (the PDF and its .docx share a stem) and fail-closed, plus by the md5 of every image embedded in an excluded file (a renamed copy is caught).
- Five security-review findings were fixed during this work (exclusion fail-open x2, PII scan bypass x2 including one regression, URL credentials in manifests, manifest-driven embedder rebuild); see the commit log.

#### Results (`outputs/reports/retrieval_report.md`, 30 questions, 2 manuals)

Grid: 6 embedders x `--max-tokens` {200, 400, 600} x `--tables` {keep, flatten} x context header on/off = 72 configurations. Best per model:

| configuration | R@1 | R@3 | R@5 | MRR | role acc | ms/query |
|---|---|---|---|---|---|---|
| WangchanX-Legal-ThaiCCL-Retriever · 400 · keep · header | 0.69 | 0.96 | 0.96 | 0.81 | 0.92 | 116 |
| bge-m3 · 400 · keep · header | 0.69 | 0.96 | 0.96 | 0.79 | 0.96 | 126 |
| e5-small · 600 · flatten · header | 0.62 | 0.85 | 0.92 | 0.72 | 0.88 | 14 |
| e5-base · 400 · flatten · header | 0.54 | 0.77 | 0.92 | 0.67 | 0.92 | 37 |
| fake (hash baseline) · 600 · keep · header | 0.38 | 0.62 | 0.65 | 0.50 | 0.64 | 0 |
| simcse-m-bert-thai · 400 · flatten · header | 0.31 | 0.54 | 0.65 | 0.42 | 0.76 | 34 |

Per question type (best configuration): text R@1 0.64 / MRR 0.79 (bge-m3 · 400 · keep · header); screenshot-only R@1 **1.00** (bge-m3 · 400 · keep · header): the OCR text pays off; cross-role MRR 0.83 (WangchanX · 400 · keep · header). One answerable question = 0.04 here (26 answerable), so WangchanX vs bge-m3 is a tie.

- **Context header:** on, in every model by several questions (bge-m3 · 400 · keep: MRR 0.79 vs 0.69 without; role accuracy 0.96 vs 0.84).
- **Chunk size:** 400 (bge-m3: 0.79 vs 0.71 at 200; 600 level with 400).
- **Tables:** keep (level with or slightly ahead of flatten; keeps the source form).
- **Model: BAAI/bge-m3** (now `DEFAULT_MODEL`). Tied best, general-purpose (WangchanX is tuned for corporate/commercial law), best role accuracy, MIT. **Low-RAM fallback: intfloat/multilingual-e5-small** (MIT): about 3 questions lower MRR, half the memory, ~9x faster queries, and its "not found" threshold (0.86) separated all 4 no-answer questions without refusing an answerable one.
- **"Not found" threshold, bge-m3:** 0.56 refuses all 4 no-answer questions but also 1 answerable one (the 48-hour question scores 0.52). Not yet safe to use as a gate.
- The thai SimCSE model is below the hashing baseline on MRR: unsupervised Wikipedia SimCSE does not suit this retrieval task.

#### Measured RAM and time (Mac, Apple M4 Pro, 26 GB; Windows not measured)

Embedding the 25 chunks with the cache off, including model loading (`/usr/bin/time -l rag-ingest embed --no-cache`):

| model | peak RAM | wall time | query (grid) |
|---|---|---|---|
| e5-small | 1.25 GB | 9.5 s | 14 ms |
| e5-base | 1.57 GB | 9.0 s | 37 ms |
| bge-m3 | 2.38 GB | 12.9 s | 126 ms |
| WangchanX | 2.38 GB | 14.2 s | 116 ms |
| simcse-m-bert-thai | 1.27 GB | 10.0 s | 34 ms |

Whole six-model grid: 175 s, peak 3.09 GB (one model loaded at a time). Downloads: e5-base 1.13 GB, bge-m3 2.27 GB (already cached), WangchanX 2.29 GB (~23 min; the Hugging Face Xet transfer failed once and was retried), SimCSE 0.72 GB.

#### Checks
- 139 tests (no model, no network); max cognitive complexity 9; no repeated literals.
- `ingest`, `benchmark` and `compare` outputs byte-identical to before this branch (the two original manuals).
- `rag-ingest chunk && rag-ingest embed --embedder local --model BAAI/bge-m3 && rag-ingest eval-retrieval` runs end to end (exit 0). The single-index report goes to `outputs/reports/retrieval_report_index.md` so it does not overwrite the grid report.
- No chunk, vector, metadata or report file contains text of the excluded manual; the PII rule finds 0 IDs in the included documents (2 phone flags).

#### Still open
- **Windows:** RAM and time not measured on the Windows CPU machine; bge-m3 needs ~2.4 GB there plus the OS. If that is too much, use e5-small.
- **Evaluation set** is small (30 questions, 2 manuals); grow it, especially no-answer questions, before trusting a threshold.
- **Names** in screenshots (e.g. user names in the system's header bar) are personal data the PII rule does not cover (it covers national IDs and phone numbers).
- `outputs/ingest/` has not been re-run for the 3 new PDFs and the 2 Word files; chunk/embed work on any set of `.md` files once it is.
- The committed `outputs/ingest/manifest.json` lacks two `rejected` records that a fresh ingest writes (ingest records rejections only when it does not hit its own OCR cache).
- The other test files still contain 28 composite assertions (Sonar S9073); `test_chunking.py` is fixed.

### 2.18 OCR ground truth from a blank box, benchmark rerun, engine choice (2026-10-20, branch `feature/ocr-ground-truth`)

#### Ground truth
- **Scope:** 18 images, all unique screenshots that hold text, per `outputs/ingest/manifest.json` (status `ocr`/`needs_ocr`). The 7 icons and 2 byte-identical duplicates are not scored.
- **Storage:** `outputs/ground_truth/<image>.txt` plus `provenance.json`, which records who made each transcription, how, and when (`ground_truth.py`).
- **By a person (3):** the earlier corrections `ผู้จัดการสาขา` p8_3, p10_1 and p11_1. They were *edited from the baseline `vision` output*, so they lean towards that engine. The 4th correction (p2_1) is an icon and is not scored.
- **By Claude (15):** each one was typed from the image alone (zoomed crops for small text), without any model output shown. The exception is the first ~170 characters of each screenshot's ingest text, which Claude had seen earlier in the session. **These have not been checked by a person.** The report gives them a separate table.
- **Conventions:**
  - text only: no radio/checkbox marks, bullets or icons;
  - clipped text only when its identity is certain;
  - screens read sidebar first, then the main panel top to bottom;
  - tables one row per line, header first;
  - step indicators as `N / ขั้นตอนที่ N / description`.
- **`rag-ingest review --blank`:** a person can make or check ground truth with it.
  - It shows each benchmarked image with an empty box: no model text, and no Claude transcription.
  - Saving writes `outputs/ground_truth/` with provenance `human` / `blank box`, which then replaces Claude's version.

#### Benchmark changes
- The report has CER tables:
  - per provenance (all / by a person / by Claude);
  - per size (small crops / screens).
- A **failure table** counts, over every benchmarked image, the outputs ingest would reject (quality rules) or that are missing. Failures matter more than a few CER points: a rejected screenshot loses all its text.
- **New quality rule `repetition_loop`:** ≥20 non-empty lines with under 25% distinct lines.
  - It catches `vision_both`'s runaway loops (3% distinct lines); real outputs have ≥52% distinct lines.
  - Ingest output is unchanged.
- **New `pipeline` row:** what ingest actually takes, i.e. `DEFAULT_SOURCES` in order, the first output that passes the quality rules. This measures the engine choice as it is used.
- **Folder discovery:** any `outputs/vision_*` folder is now benchmarked without editing `METHODS`.

#### Results (18 images, mean CER, lower is better)

| method | all (18) | by a person (3) | by Claude (15) | small crops (3) | screens (15) | failures /18 |
|---|---|---|---|---|---|---|
| **pipeline** (typhoon → hint) | **0.164** | 0.100 | **0.177** | 0.034 | **0.190** | **0** |
| vision_typhoon | 0.225 | **0.100** | 0.251 | 0.034 | 0.264 | 2 (prompt echo, "no text") |
| vision_hint | 0.228 | 0.182 | 0.237 | **0.000** | 0.274 | 0 |
| tesseract | 0.291 | 0.372 | 0.274 | 0.237 | 0.301 | 0 |
| vision_both | 0.292 | 0.204 | 0.310 | 0.000 | 0.351 | 1 (loop) |
| ocr (PaddleOCR) | 0.319 | 0.244 | 0.333 | 0.013 | 0.380 | 0 |
| vision (baseline) | 0.397 | 0.133 | 0.450 | 0.054 | 0.466 | 1 ("no text") |

Pooled CER (all 18): pipeline 0.169, typhoon 0.265, tesseract 0.288, hint 0.298. Full tables: `outputs/reports/benchmark_report.md`, `outputs/reports/benchmark.csv`.

#### What the numbers say
- **Typhoon reads large screens and tables best** (e.g. CER 0.02–0.09 on `ผู้จัดการเขต` p3_1, p4_1 and `ผู้จัดการสาขา` p5_1, where the local models score 0.11–1.00).
- **Typhoon fails outright on 2 of 18:**
  - p7_1 echoed the prompt (CER 0.99);
  - p7_3 answered "no readable text" on a real screen (0.93).
- **vision_hint never failed** and is exact on small crops.
- The fallback chain takes Typhoon's strength and covers both of its failures with vision_hint. That makes it clearly the best (0.164 mean vs 0.225 / 0.228 for either alone).
- **The baseline `vision` looked second-best on the 3 corrections made by a person (0.133) but is last on the 15 blank-box ones (0.450).** This shows the anchoring bias: those corrections were edited from its output. It is why ground truth should come from a blank box.
- **Tesseract is a respectable third on screens** (and fast, no GPU), but weak on small crops (0.237).

#### Decision: main engine
**Main OCR engine: Typhoon OCR (`vision_typhoon`), with `vision_hint` (local qwen2.5vl:3b + hint) as the automatic fallback.** This is the existing `ingest.DEFAULT_SOURCES = ("vision_typhoon", "vision_hint")`, now backed by a benchmark, so the code does not change.
- **Why not one engine alone:**
  - Typhoon alone loses 2 screenshots entirely;
  - vision_hint alone is ~0.07 CER worse and much worse on long tables.
- **Why not Tesseract or PaddleOCR as the fallback:** both have higher CER than vision_hint (0.291 and 0.319), and PaddleOCR is very poor on some screens (0.94 on `ผู้จัดการเขต` p2_1).
- **Condition:** Typhoon is an API. If documents may not leave the internal network and Typhoon is not hosted internally, use **`vision_hint` alone** (0 failures, 0.228). In that case, prompt tuning and tiling (step 3) are the levers to close the gap on long screens.

#### Caveats
- 15 of 18 references are Claude's reading, not a person's, and 18 images are still few. Treat the gaps between methods within ~0.03 as ties (typhoon vs hint alone; tesseract vs vision_both).
- CER ignores layout (HTML, markdown and whitespace are stripped), so it does not judge Typhoon's HTML tables against plain text. That is the open table-format question (step 2).
- Two references score poorly for every engine (`ผู้จัดการสาขา` p7_3, `ผู้จัดการเขต` p5_1). They may reflect transcription conventions as much as OCR errors; a person should look at these first.

#### Still open
- A person should check the 15 Claude transcriptions with `rag-ingest review --blank` (start with p7_3 and `ผู้จัดการเขต` p5_1). Then rerun `rag-ingest benchmark`.
- Invented foreign-language words (hallucinated non-Thai tokens) are still not detected automatically. Loops now are.

### 2.19 Reports in their own folder (2026-10-20, branch `feature/outputs-reports`)
- All reports moved from the top of `outputs/` to `outputs/reports/` (`Paths.reports` in `config.py`): `benchmark_report.md`, `benchmark.csv`, `compare_*.md` and `retrieval_report*.md/.csv`. The commands write there by default and create the folder when needed; `--report`/`--csv` still override.
- The two plain-text dumps from the initial commit (`ผู้จัดการสาขา.txt`, `ผู้จัดการเขต.txt`) moved to `outputs/text/`. The pipeline does not use them; delete them if they are not needed.

### 2.20 Modules grouped by stage (2026-10-20, branch `refactor/group-modules`)
- `ocr_eval/`: `ground_truth`, `benchmark`, `compare`, `metrics` (moved, names kept).
- `retrieval/`: `indexing`, `vectorstore`, `index_commands`, `eval_commands`, plus `retrieval_eval` -> `retrieval/evaluation.py` and `retrieval_report` -> `retrieval/report.py` (the package name already says "retrieval").
- Moves only, no behaviour change: 157 tests pass, `rag-ingest benchmark` writes an identical report, and `rag-ingest search` works on the existing bge-m3 index. Imports are now e.g. `from rag_ingest.ocr_eval.benchmark import ...`, `from rag_ingest.retrieval.indexing import ...`.

### 2.21 Table format, and chunking/embedding to match (2026-10-20, branch `feature/table-format`)

#### Question
Typhoon writes tables as HTML (`<table><tr><td>`) and sidebar items as Markdown headings (`### 1.บันทึกอุบัติเหตุ`). Should that HTML be kept, or turned into plain text, and how should chunking and embedding handle it?

#### What the tables are
- The two ingested manuals contain 3 tables, all HTML from Typhoon and all inside screenshots of list screens:
  - `ผู้จัดการสาขา` p6 "ติดตามสถานะ" (7 rows × 10 columns);
  - `ผู้จัดการเขต` p3 "ติดตามสถานะ" (10 × 8);
  - `ผู้จัดการเขต` p4 "อนุมัติอุบัติเหตุ" (10 × 9).
- **Their rows are sample records:** employee names, branches, dates and statuses. None of the 30 evaluation questions asks about a row. They ask what a screen is for and what its statuses mean, which the manual's text explains next to the screenshot.
- **With `keep`, the rows took 7 of 25 chunks.** Some chunks (e.g. `ผู้จัดการเขต#p4#c07`) were almost only sample rows, and the explanation (how to approve or reject) was pushed into the following chunk.
- **The rows held 22 mentions of employee names** in chunks and vectors.
- **The tables are parsed correctly** (`chunking/tables.py`, colspan kept aligned). The issue was what to do with the rows, not the HTML.

#### Decisions
1. **Ingest keeps Typhoon's raw HTML.** `outputs/ingest/*.md` is the lossless record. Chunks cite character ranges in it, and the OCR cache matches its text, so converting at ingest would lose structure and break both.
2. **One table format after ingest: Markdown.**
   - Typhoon's HTML is converted at chunk time; Word tables already are Markdown.
   - The display text of a screenshot (what a search hit shows, and what the PII guard scans) is now the same rendering as the embedded text, inside the screenshot tags, instead of the raw HTML.
3. **Tables inside screenshots get their own setting:** `--screenshot-tables` (`ChunkSettings.screenshot_tables`).
   - **Default `columns`:** the header row only, as a Markdown table without rows. `keep` and `flatten` remain available.
   - Tables in a document's own text keep `--tables` (default `keep`), because in Word files the rows are real content.
4. **Screenshot OCR is tidied** (`text.tidy_ocr`): Markdown heading marks and checkbox glyphs (☐ ☑ ☒ ✓ ✗) are removed, in lines and in table cells. Other text is untouched.

#### Measurement (30 questions, 400 tokens, context header; scratch run of all 12 combinations)

| model | OCR tidy | screenshot tables | chunks | R@1 | R@3 | MRR | role acc | MRR on the 11 table-page questions |
|---|---|---|---|---|---|---|---|---|
| bge-m3 | no | keep (before) | 25 | 0.69 | 0.96 | 0.79 | 0.96 | 0.82 |
| bge-m3 | no | flatten | 25 | 0.58 | 0.92 | 0.72 | 0.92 | 0.65 |
| bge-m3 | no | columns | 19 | 0.65 | 0.88 | 0.79 | 0.92 | 0.75 |
| bge-m3 | yes | keep | 22 | 0.54 | 0.92 | 0.71 | 0.80 | 0.67 |
| bge-m3 | yes | flatten | 21 | 0.62 | 0.96 | 0.77 | 0.92 | 0.70 |
| **bge-m3** | **yes** | **columns (now)** | **18** | **0.69** | 0.88 | **0.80** | **0.96** | 0.80 |
| e5-small | no | keep (before) | 25 | 0.54 | 0.85 | 0.67 | 0.88 | 0.64 |
| e5-small | no | flatten | 25 | 0.50 | 0.77 | 0.63 | 0.80 | 0.46 |
| e5-small | no | columns | 19 | 0.65 | 0.85 | 0.75 | 0.92 | 0.76 |
| e5-small | yes | keep | 22 | 0.50 | 0.85 | 0.66 | 0.80 | 0.58 |
| e5-small | yes | flatten | 21 | 0.65 | 0.85 | 0.75 | 0.88 | 0.66 |
| **e5-small** | **yes** | **columns (now)** | **18** | **0.69** | 0.85 | **0.77** | **0.92** | **0.80** |

#### How to read it
- **One answerable question is 0.04.** The per-question ranks show changes spread over questions that have nothing to do with tables (q03, q08, q15, q23): shortening a screenshot moves chunk boundaries and reshuffles a question or two either way. Gaps of about 0.04 are noise.
- **What holds across every run:**
  - `columns` is never worse than `keep` overall: a tie on bge-m3 (MRR 0.80 vs 0.79), clearly better on e5-small (0.77 vs 0.67). On the 11 table-page questions bge-m3 is level too (0.80 vs 0.82, half a question).
  - `columns` is stable whether the cleanup is on or off, while `keep` swings by 0.08.
  - `flatten` is the worst on the table-page questions for both models.
- **The "not found" threshold is unaffected:** bge-m3 0.57, false answers 0.00, 1 answerable question refused, as before.
- **The real pipeline reproduces the scratch run:** `rag-ingest chunk && rag-ingest embed && rag-ingest eval-retrieval` gives 18 chunks, R@1 0.69, R@3 0.88, MRR 0.80, role accuracy 0.96 (`outputs/reports/retrieval_report_index.md`).
- **Chunks: 25 → 18.** The approve screen (`ผู้จัดการเขต` p4) went from 3 chunks of mostly sample rows to one 121-token chunk with the screen's labels, the column names and the how-to-approve text together.
- **Names from the three tables in chunks: 22 → 0.**

#### Full grid (`outputs/reports/retrieval_report.md`, 108 configurations)
The grid's table dimension is now `--screenshot-tables`: 6 models × {200, 400, 600} tokens × {keep, flatten, columns} × context header on/off, with the OCR tidy on. The old grid's `--tables` dimension was measuring the same screenshot tables, because the manuals have no tables of their own.

Best configuration per model:

| configuration | chunks | R@1 | R@3 | MRR | role acc | threshold | false answer | answerable refused |
|---|---|---|---|---|---|---|---|---|
| bge-m3 · 600 · flatten · header | 20 | 0.81 | 0.88 | 0.84 | 0.96 | 0.57 | 0.00 | 0.04 |
| WangchanX · 600 · keep · header | 19 | 0.77 | 0.88 | 0.84 | 0.96 | 0.55 | 0.00 | 0.04 |
| e5-small · 400 · columns · header | 18 | 0.69 | 0.85 | 0.77 | 0.92 | 0.86 | 0.00 | 0.04 |
| e5-base · 600 · flatten · header | 19 | 0.58 | 0.85 | 0.69 | 0.80 | 0.84 | 0.00 | 0.19 |
| fake (hash baseline) · 200 · flatten · header | 36 | 0.35 | 0.58 | 0.46 | 0.60 | 0.21 | 0.25 | 0.04 |
| simcse-m-bert-thai · 200 · flatten · header | 56 | 0.35 | 0.50 | 0.45 | 0.60 | 0.30 | 0.00 | 0.15 |

Every real model's best MRR is level with or above the previous grid (2.17): bge-m3 0.79 → 0.84, WangchanX 0.81 → 0.84, e5-small 0.72 → 0.77, e5-base 0.67 → 0.69.

Mean MRR over the three chunk sizes, context header on:

| model | keep | flatten | columns |
|---|---|---|---|
| bge-m3 | 0.76 | **0.79** | 0.77 |
| WangchanX | **0.78** | 0.77 | **0.78** |
| e5-small | 0.67 | 0.68 | **0.73** |
| e5-base | 0.65 | 0.65 | **0.67** |

#### Why `columns` stays the default
- **On the strong models, retrieval does not separate the modes.**
  - bge-m3's single best cell (600 · flatten, 0.84) is one question above 400 · columns (0.80).
  - Its mean over sizes favours flatten by 0.02; WangchanX is a tie.
- **`columns` is clearly best on the small models,** including the low-memory fallback e5-small (0.73 against 0.67–0.68).
- **Beyond retrieval:**
  - fewest chunks (18 at 400 tokens);
  - each screen's explanation stays in the same chunk as the screen;
  - no sample records, so the 22 mentions of employee names from the tables are gone from chunks and vectors.
- **Chunk size stays at 400.** With `columns`, 400 and 600 tie for both bge-m3 (0.80 / 0.80) and e5-small (0.77 / 0.77), and smaller chunks cite more precisely.

#### Embedding
Unchanged: bge-m3 (`DEFAULT_MODEL`, tied with WangchanX again, best role accuracy), with e5-small as the low-memory fallback. The new format helps the fallback most (MRR 0.67 → 0.77 at the default settings), which narrows its gap to bge-m3 from 0.12 to 0.03.

#### Still open
- **Names outside tables:** screenshots still carry names in forms and in the header bar (`ชื่อ : อิทธิฤทธิ์ มีชัย`, the logged-in user). `columns` does not cover them; the open question of whether the PII rule should cover names in screenshots remains.
- **Screenshots of reference tables** (e.g. a table of fees or rights captured as an image) would lose their rows under `columns`. None exist in the ingested manuals; use `--screenshot-tables keep` for such documents, or revisit if the new PDFs/Word files have them once they are ingested.

### 2.22 Databases: PostgreSQL + Qdrant (2026-10-22, branch `feature/databases`)

#### What was built
- `docker-compose.yml`: `postgres:17.6` and `qdrant/qdrant:v1.15.4`, named data volumes, ports bound to 127.0.0.1. Credentials only from `.env` (git-ignored); `.env.example` lists the variables. No password default anywhere: a missing `POSTGRES_PASSWORD` stops the command.
- **PostgreSQL schema** (`databases/migrations/001_documents_chunks.sql`, applied once each, recorded in `schema_migrations`):
  - `documents`: id (the document stem, as in chunk ids), name, source path (the ingest .md), content hash (its sha256 = the chunks' `source_sha256`), role, doc type, status (`verified`/`unverified`, default unverified), ingested time.
  - `chunks`: chunk id, index run, document id (FK, cascade), section, page start/end, type, text (`display_text`, what an answer cites), token count, embedding model. Primary key (index run, chunk id), so two runs can be loaded side by side.
  - No ACL or conversation tables (other parts); role and status are there for them.
- **Qdrant:** one collection per index run, named after the run folder; cosine, dim from the index (1024 for BGE-M3); keyword payload indexes on document, role, doc_type. Point id = uuid5(fixed namespace, chunk id). Payload: chunk id, document, role, doc type, kind, type, section, page start/end. No chunk text in Qdrant.
- **Code:** `QdrantVectorStore` implements the existing `VectorStore` protocol (add, save as a no-op, search with the same filters) plus count, delete by document, and points (used as the undo copy). `PostgresChunkStore` for the two tables. `MemoryChunkStore` mirrors it in memory, including transaction rollback. `NumpyVectorStore` is unchanged and still the index format and test backend.
- `rag-ingest load [--index RUN] [--status ...] [--dry-run]`, and `rag-ingest eval-retrieval --backend qdrant`.

#### Decisions
- **`type` is derived, not new data.** Chunks have `kind` text/ocr/mixed. `type` = `screenshot` when the chunk holds any OCR'd screenshot (ocr or mixed), `table` when its display text has a Markdown table separator row, else `text`. `kind` stays in the Qdrant payload.
- **Pages:** chunks span `page_start`–`page_end`; both are stored (the evaluation needs the range).
- **Idempotency:** skip when the stored content hash and status match and both stores hold exactly the document's chunk ids for this run (so a store that drifted is repaired by the next load). A changed hash replaces the document's chunks in every run: PostgreSQL deletes them, and the loader removes its points from the older runs' collections.
- **All-or-nothing, per document:** PostgreSQL rows are written inside a transaction; the Qdrant upsert and the removal of the document's other points happen inside it; then commit. Any exception (Qdrant down, commit fails) rolls back and puts the document's previous Qdrant points back from the copy taken first. A failed document is reported, the others still load, exit code 1.
- **Safety checks carry over and fail closed:** `load_index` (stale check + manifest allow-list) runs first; then every vector row must be a chunk of the current `chunks.jsonl`, not in `rejected.jsonl`, from a recorded source, and not excluded (`EXCLUDED_DOCUMENTS` by stem + type, an `unknown` type counts as excluded, plus the chunk manifest's `excluded`). Any hit refuses the whole load before anything is written.
- **Exact search** (`SearchParams(exact=True)`): at this size HNSW would also be exact, but parity should not depend on it.

#### Checks
- `tests/test_databases.py`: 22 tests (40 cases); those that touch a store run on the in-memory stand-ins (`MemoryChunkStore` + `QdrantClient(":memory:")`, always) and on the docker databases (throwaway schema and collections; skipped when not reachable). Covers migrations, both stores written and linked, payload without text, Qdrant vs numpy search with filters, skip on reload, a changed document replacing its chunks (this run and older runs), status kept, Qdrant failing before/after upsert and on delete, the PostgreSQL commit failing after Qdrant was written, one failed document not stopping the others, a real Thai ID chunk never reaching either store, an index with a rejected chunk refused, the excluded PDF (real stem, pdf source) never loaded, an old index holding it refused, stale and untrusted indexes refused, the CLI.
- **Parity gate** (`tests/test_qdrant_parity.py`): the committed chunks' BGE-M3 index (`BAAI-bge-m3__1a6e93b0`, 18 chunks) loaded into Qdrant and searched with plain dense search: the top 5 is identical to numpy for all 30 questions (scores within 1e-4), **R@5 0.96, MRR 0.80**, same as numpy. Set `RAG_PARITY_DOCKER=1` to repeat it on the docker Qdrant.
- `rag-ingest load --dry-run` on the real index: 2 documents, 18 chunks (ผู้จัดการสาขา 12, ผู้จัดการเขต 6).

#### Docker gotchas
- An **empty `QDRANT_API_KEY`** makes Qdrant require a key nobody has (every request 401). Compose now refuses to start without one; set it in `.env`.
- The client (1.19) is newer than the pinned server image (1.15.4): the client's compatibility check is switched off (`check_compatibility=False`); the calls used are basic and the tests pass.

#### Still open
- **Docker run (2026-10-22):** after Docker Desktop was restarted, all 40 cases of `tests/test_databases.py` passed on the real databases, the parity test passed on the docker Qdrant (`RAG_PARITY_DOCKER=1`), and `rag-ingest load` loaded 2 documents / 18 chunks, then skipped both on the second run. `eval-retrieval --backend qdrant` gives R@1 0.69, R@3 0.88, R@5 0.96, MRR 0.80.
- A document removed from the corpus is not deleted from the databases by `load` (it only adds/replaces); `delete_document` exists on both stores for that.
- Hybrid search, reranking, ACL and conversation tables are out of scope here.

### 2.5 Findings
- The `--hint` is applied unevenly: in 3 of the first 6 hint results `paddle_hint` was `false`, because no PaddleOCR line reached confidence 0.8. Those images effectively got only temperature 0 + upscale.
- PaddleOCR output is poor on some images (e.g. first sample had confidences 0.36–0.72 with garbage text), so it is a weak hint source.
- Tesseract (see 2.6) is a good second opinion where it reads text, but not a replacement for the vision model: 7/27 images came back empty, and it reads far fewer lines on large screenshots.

---

## 3. What has NOT been done

- `review_vision.py` has never been opened in a browser; only confirmed to compile.
- `--hint` and `--upscale` were only exercised through the batch run; no controlled test of each flag alone.
- ~~Only 4 corrected transcriptions~~ -> 18 images now have ground truth (see 2.18), but 15 of them are Claude's blank-box readings, not yet checked by a person.
- Tesseract output is not yet used as a hint for the vision model, and not yet included in `compare_vision.py` (which only compares two vision folders).
- No controlled test of Tesseract settings (`--psm`, upscaling the tiny crops that came back empty).
- Image tiling (splitting large images) not implemented.
- Fine-tuning not started.
- Comparison does not yet show which images actually received a hint.

---

## 4. What will be done (planned order)

1. ~~Finish the hint run~~ — done (27/27, context fix `NUM_CTX = 8192`).
2. ~~Rerun the comparison on all 27~~ — done (see 2.4).
3. ~~Add Tesseract~~ — done (see 2.6). Follow-ups: try Tesseract lines as a second/alternative hint for the vision model (add a `--hint-source` option); try `--psm 6`/upscaling on the 7 empty crops; extend the comparison to include PaddleOCR and Tesseract.
4. ~~**Manual review**: correct 10-15+ more images, rerun the benchmark, choose the engine~~ — done (see 2.18: Typhoon with vision_hint fallback). Still open: a person checks the Claude transcriptions (`rag-ingest review --blank`); decide how to handle Typhoon's HTML/markdown output.
5. **Improve the comparison**: add a column showing whether the hint was applied per image.
6. **Optional next levers**: prompt tuning (edit `prompts/thai_document_ocr.txt`, compare), image tiling for large images, combining PaddleOCR/Tesseract disagreement flags.
7. **Fine-tuning (only if the above is not enough)**: LoRA/QLoRA of Qwen2.5-VL (Unsloth / LLaMA-Factory / ms-swift) on the corrected image-text pairs, convert to GGUF, load into Ollama. Needs a few hundred or more corrected examples; risk of hurting other content if data is thin.

---

## 5. Gotchas

- Use `.venv\Scripts\python.exe` (or activate the venv); system Python will fail on `import ollama`.
- Install packages with uv, not pip.
- Background run logs are buffered; check progress by counting files in the output folder.
- Upscaling raises the image token count; with Ollama's default 4096 context this makes larger images fail (HTTP 400).
- CPU inference is slow (minutes per image) and memory-heavy; do not run two batches at once.
