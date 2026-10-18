# rag-ingestion-poc — Working Notes

Last updated: 2026-10-16 (Tesseract, both-hint run 27/27, Typhoon API 27/27, benchmark script)

Goal: extract text from Thai PDFs (embedded images) accurately enough for RAG ingestion, using only local tools.

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
