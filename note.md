# rag-ingestion-poc — Working Notes

Last updated: 2026-10-16 (after first A/B comparison)

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

`requirements.txt`: PyMuPDF, ollama, pillow.

Folders: `prompts/` (OCR prompt), `outputs/images/` (27 extracted images), `outputs/ocr/` (PaddleOCR results), `outputs/vision/` (baseline vision results), `outputs/vision_hint/` (hint + upscale results).

---

## 2. What has been done

### 2.1 `src/run_vision_ocr.py` (vision OCR runner)
Sends the image plus `prompts/thai_document_ocr.txt` to `qwen2.5vl:3b`; saves one JSON per image (`image`, `model`, `prompt_file`, `settings`, `text`, `review_status: pending`).

Options:
- `--image PATH` — one image.
- `--all [DIR]` — batch over a folder (default `outputs/images/`). A failing image does not stop the run; prints a done/skipped/failed summary.
- `--skip-existing` — skip images that already have a result JSON (resume).
- `--hint` — adds PaddleOCR lines with confidence >= 0.8 from `outputs/ocr/<name>.txt` to the prompt as a reference the model is told not to fully trust.
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
- **Hint + upscale `outputs/vision_hint`**: **16/27 saved**. First attempt was stopped automatically by Claude Code (system memory critically low). The resume run (`--skip-existing`) then processed all 27 but **11 images FAILED** with `request (~4200-4460 tokens) exceeds the available context size (4096 tokens)` (HTTP 400). Cause: upscaling + the hint text make the image/prompt exceed Ollama's default 4096 context for `qwen2.5vl:3b`. The failed ones are mostly larger/denser images (e.g. `ผู้จัดการสาขา_page_11_image_1`, `page_3_image_2`, `page_4_image_2`, `page_6_image_1`, `page_7_image_1`, `page_8_image_1`, `page_9_image_1` and the `ผู้จัดการเขต` page_4/5 images). This was only seen for the first 23 of 27 at last check; the run may still have been working on the rest.
- **First comparison** (`src/compare_vision.py --diff`, 16 images common to both folders, saved to `outputs/compare_report.md`):
  - 6/16 identical (the short ones); mean A~B similarity 0.70.
  - B (hint + upscale + temp 0) looked better by reading the diffs: removed a 3,298-char repetition loop on `ผู้จัดการเขต_page_3_image_1` (503 chars clean in B); fewer invented words (e.g. A "ลงบัมมูล" vs B "ลบข้อมูล"); more complete text on `page_4_image_1` and `page_4_image_3`.
  - B possible regressions / unclear: `page_5_image_1` shrank 1834 -> 848 chars and lost real content; "รายงานการขออนุมัติ" (B) vs "รายงานการอุบัติเหตุ" (A) on `ผู้จัดการเขต_page_2_image_1`; "ตกลง!" vs "คลิก!"; heading "ส่วน" dropped on `page_4_image_1`. Names/numbers still misread in both.
  - Caveat: judged by reading Thai in the diffs, **not verified against images**; no ground truth, so no CER. Baseline also ran without temperature 0, so the gain cannot be split between the hint, upscale and temperature.
- Sanity check on the first image: baseline transcription of `ผู้จัดการเขต_page_2_image_1.png` was correct.

### 2.5 Findings
- The `--hint` is applied unevenly: in 3 of the first 6 hint results `paddle_hint` was `false`, because no PaddleOCR line reached confidence 0.8. Those images effectively got only temperature 0 + upscale.
- PaddleOCR output is poor on some images (e.g. first sample had confidences 0.36–0.72 with garbage text), so it is a weak hint source.
- Tesseract is **not** in the project (no code, no `pytesseract`, no install).

---

## 3. What has NOT been done

- Hint + upscale run is incomplete: 11 images failed on the 4096-token context limit (see 2.4). Fix not implemented yet.
- Comparison covers only 16 of 27 images.
- `review_vision.py` has never been opened in a browser; only confirmed to compile.
- `--hint` and `--upscale` were only exercised through the batch run; no controlled test of each flag alone.
- No corrected transcriptions exist yet -> no CER / real accuracy numbers.
- Tesseract not added.
- Image tiling (splitting large images) not implemented.
- Fine-tuning not started.
- `outputs/compare_report.md` does not exist yet.
- Comparison does not yet show which images actually received a hint.

---

## 4. What will be done (planned order)

1. **Fix the context-size failures first** (11 images failed with the 4096-token limit): pass `options={"num_ctx": 8192}` to `ollama.chat` (more memory, slower on CPU), or lower `--upscale` / cap the max image side, or shorten the hint. Then **finish the hint run** (close heavy programs first to avoid the memory kill):
   ```powershell
   .\.venv\Scripts\python.exe .\src\run_vision_ocr.py --all --hint --upscale 1000 --output-dir outputs/vision_hint --skip-existing
   ```
   Only restart when memory allows (a partial comparison on the 16 existing images is possible in the meantime).
2. **Rerun the comparison on all 27**: `python src/compare_vision.py --diff` and read `outputs/compare_report.md` (already run once on 16 images, see 2.4).
3. **Add Tesseract** (approved, to be done after the comparison):
   - Install the Tesseract program for Windows with the Thai (`tha`) data.
   - `uv pip install --python .venv/Scripts/python.exe pytesseract`, add to `requirements.txt`.
   - New batch script writing output in the same `text<TAB>confidence=` format as the PaddleOCR files, so it can serve as a second hint source or a third engine to compare.
4. **Manual review** of a sample (or all) of the results with `review_vision.py` to get `corrected_text`; rerun the comparison to get CER for each approach.
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
