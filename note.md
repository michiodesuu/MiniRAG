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
Compares two result folders (default `outputs/vision` vs `outputs/vision_hint`): per-image similarity, text lengths, optional line diff (`--diff`), report saved to `outputs/compare_report.md`. If `corrected_text` exists it also reports character error rate (CER) for both runs.

### 2.4 Runs
- **Baseline `outputs/vision`**: complete, 27/27 images, 0 failures. Settings: no hint, no upscale. (Started before temperature 0 was added, so it used Ollama's default temperature.)
- **Hint + upscale `outputs/vision_hint`** (`--hint --upscale 1000`, temp 0): **27/27 saved, 0 failures** (final rerun of the last 3 images completed).
  - Attempt 1 was stopped by Claude Code (system memory critically low) after 16 images.
  - Resume run: **11 images FAILED** with `request (~4200-4460 tokens) exceeds the available context size (4096 tokens)` (HTTP 400). Cause: upscaling + hint text exceed Ollama's default 4096 context for `qwen2.5vl:3b`.
  - **Fix applied**: `NUM_CTX = 8192` in `src/run_vision_ocr.py` (passed as `options.num_ctx`, recorded in each JSON's `settings`). Rerun succeeded for 4 of the failed images (`สาขา_page_11_image_1`, `page_3_image_2`, `page_4_image_2`, `page_6_image_1`); that rerun was then stopped again for low memory. (The 4 `ผู้จัดการเขต` page_4/5 images had succeeded in the resume run itself.)
  - Final rerun: `สาขา_page_7_image_1`, `page_8_image_1`, `page_9_image_1` completed OK with the 8192 context.
  - Note: results from before the fix used the default context (4096), the later ones 8192; the mix is unlikely to change transcription much but is not controlled.
- **Comparison** (`src/compare_vision.py --diff`, saved to `outputs/compare_report.md`), latest on **all 27 images**: 8/27 identical (the short ones), mean A~B similarity 0.58.
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
- **Typhoon OCR API** (company internal, OpenAI-compatible): `POST http://ocr-api.sustainportal.muangthaicap.local/v1/chat/completions`, model `typhoon-ai/typhoon-ocr1.5-2b`. Reachable from this PC (resolves to 10.0.1.11), no API key needed. Added `--backend api --api-url ... --api-model ...` to `run_vision_ocr.py` (key only via env `OCR_API_KEY`; URL/model also via `OCR_API_URL` / `OCR_API_MODEL`; internal URL intentionally not hardcoded). Batch of 27 finished in a few minutes -> `outputs/vision_typhoon/` (plain prompt, no hint, no upscale), 0 failures.
  - Looks fuller/more accurate on big screens (e.g. `เขต_page_3_image_1`: 1805 chars vs 503; correct place names) and on clean prose; still weak on tiny crops (`เขต_page_4_image_2` returned 1 char; `สาขา_page_8_image_2` gave `[ถังขยะรีไซเคิล]`).
  - Output contains HTML tables (`<table><tr><td>`) and markdown (`# `, `* `): decide whether to keep (structure) or strip to plain text before chunking. Comparison with hint run: `outputs/compare_hint_vs_typhoon.md` (mean similarity 0.63).

### 2.8 `src/benchmark.py` (evaluation)
Scores every method against hand-corrected text: vision, vision_hint, vision_both, vision_typhoon, PaddleOCR (`outputs/ocr`), Tesseract. Ground truth = `corrected_text` saved by `review_vision.py` in any `outputs/vision*` folder, or `outputs/ground_truth/<image>.txt`. Normalizes (strips HTML/markdown, ignores whitespace), character error rate capped at 1.0 per image, mean + pooled CER, empty-output count, split into short (<=60 chars) and long references. Writes `outputs/benchmark_report.md` and `outputs/benchmark.csv`. Run: `python src/benchmark.py`.
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

### 2.5 Findings
- The `--hint` is applied unevenly: in 3 of the first 6 hint results `paddle_hint` was `false`, because no PaddleOCR line reached confidence 0.8. Those images effectively got only temperature 0 + upscale.
- PaddleOCR output is poor on some images (e.g. first sample had confidences 0.36–0.72 with garbage text), so it is a weak hint source.
- Tesseract (see 2.6) is a good second opinion where it reads text, but not a replacement for the vision model: 7/27 images came back empty, and it reads far fewer lines on large screenshots.

---

## 3. What has NOT been done

- `review_vision.py` has never been opened in a browser; only confirmed to compile.
- `--hint` and `--upscale` were only exercised through the batch run; no controlled test of each flag alone.
- Only 4 corrected transcriptions exist so far -> benchmark numbers are not meaningful yet (aim for 10-15+, include long tables; correct from a blank box or a neutral folder to avoid anchoring on the baseline).
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
4. **Manual review**: correct 10-15+ more images with `review_vision.py` (especially long tables), then run `python src/benchmark.py` for real CER per method. Also decide how to handle Typhoon's HTML/markdown output.
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
