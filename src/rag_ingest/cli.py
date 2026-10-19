"""Command line: `rag-ingest <command>` (or `python -m rag_ingest <command>`)."""
import argparse
import os
import sys
from urllib.parse import urlsplit

from rag_ingest.config import ENCODING, Paths, SUPPORTED_IMAGE_EXTENSIONS


def cmd_inspect(args, paths: Paths):
    from rag_ingest.extract import inspect
    inspect(paths.resolve(args.docs_dir))


def cmd_extract_images(args, paths: Paths):
    from rag_ingest.extract import extract_images
    extract_images(paths.resolve(args.docs_dir), paths.resolve(args.output_dir))


def cmd_extract_text(args, paths: Paths):
    from rag_ingest.extract import extract_text
    extract_text(paths.resolve(args.docs_dir), paths.resolve(args.output_dir))


VISION_ENGINES = ("ollama", "api")
LOOPBACK_HOSTS = ("localhost", "127.0.0.1", "::1")


def warn_if_clear_text(url: str):
    """Images (and any API key) would cross the network unencrypted."""
    parts = urlsplit(url)
    if parts.scheme != "https" and parts.hostname not in LOOPBACK_HOSTS:
        print(f"WARNING: {parts.scheme}:// sends images unencrypted to {parts.hostname}; "
            "use https if the server supports it.", file=sys.stderr)


def api_options(args) -> dict:
    if not (args.api_url and args.api_model):
        sys.exit("--engine api needs --api-url and --api-model (or OCR_API_URL / OCR_API_MODEL)")
    warn_if_clear_text(args.api_url)
    # The key comes from the environment only, never the command line.
    return {"url": args.api_url, "model": args.api_model, "key": os.environ.get("OCR_API_KEY")}


def engine_options(args, paths: Paths) -> dict:
    for flag in ("hint", "prompt"):
        if getattr(args, flag) and args.engine not in VISION_ENGINES:
            sys.exit(f"--{flag} only applies to the vision engines (ollama, api)")
    if args.engine == "tesseract":
        return {"upscale": args.upscale, "psm": args.psm}
    if args.engine == "paddle":
        return {}
    vision = {"paths": paths, "hint_source": args.hint, "upscale": args.upscale,
              "prompt": paths.resolve(args.prompt) if args.prompt else None}
    if args.engine == "ollama":
        return {**vision, "model": args.model or "qwen2.5vl:3b", "num_ctx": args.num_ctx}
    return {**vision, **api_options(args)}


def ocr_one(engine, store, image):
    if not image.is_file():
        sys.exit(f"Image not found: {image}")
    result = engine.recognize(image)
    out = store.save(image, result)
    print(f"\n=== OCR RESULT ===\n\n{result.text}\n\nSaved to: {out}")


def ocr_folder(engine, store, folder, skip_existing: bool):
    from rag_ingest.batch import run_batch
    from rag_ingest.imaging import list_images

    if not folder.is_dir():
        sys.exit(f"Image directory not found: {folder}")
    images = list_images(folder, SUPPORTED_IMAGE_EXTENSIONS)
    if not images:
        sys.exit(f"No supported images found in {folder}")
    summary = run_batch(engine, store, images, skip_existing)
    print(f"\n{summary}\nResults in: {store.folder}", flush=True)


def cmd_ocr(args, paths: Paths):
    from rag_ingest.engines import create_engine
    from rag_ingest.stores import open_store

    engine = create_engine(args.engine, **engine_options(args, paths))
    store = open_store(paths, paths.resolve(args.output_dir or paths.method(engine.default_output)),
                    kind=engine.store_kind)
    if args.image:
        ocr_one(engine, store, paths.resolve(args.image))
    else:
        ocr_folder(engine, store, paths.resolve(args.images_dir), args.skip_existing)


def cmd_review(args, paths: Paths):
    from rag_ingest.review import serve
    from rag_ingest.stores import VisionStore
    serve(VisionStore(paths.resolve(args.dir), paths), args.port, not args.no_browser)


def cmd_compare(args, paths: Paths):
    from rag_ingest.compare import compare, render
    from rag_ingest.stores import VisionStore

    rows, only = compare(VisionStore(paths.resolve(args.a), paths),
                        VisionStore(paths.resolve(args.b), paths))
    try:
        report = render(rows, only, args.a, args.b, args.diff)
    except ValueError as error:
        sys.exit(str(error))
    print(report)
    if args.report:
        out = paths.resolve(args.report)
        out.write_text(report, encoding=ENCODING)
        print(f"\nSaved to: {out}")


def cmd_benchmark(args, paths: Paths):
    from rag_ingest import benchmark as bm
    from rag_ingest.config import METHODS
    from rag_ingest.stores import open_store

    stores = {m.name: open_store(paths, m.name, m.kind) for m in METHODS}
    truth = bm.load_ground_truth([s for s in stores.values() if s.folder.is_dir()],
                                paths.resolve(args.ground_truth_dir))
    if not truth:
        sys.exit("No ground truth yet. Correct some images with:\n"
                "  rag-ingest review --dir outputs/vision_hint\n"
                "(or put plain corrected text in outputs/ground_truth/<image>.txt), "
                "then rerun this command.")

    methods = {}
    for name in bm.method_names(args.exclude):
        if stores[name].folder.is_dir() and (texts := stores[name].texts()):
            methods[name] = texts
    names = list(methods)
    rows = bm.score_all(truth, methods)
    report = bm.render(rows, names)
    print(report)

    report_path, csv_path = paths.resolve(args.report), paths.resolve(args.csv)
    report_path.write_text(report, encoding=ENCODING)
    bm.write_csv(csv_path, rows, names)
    print(f"\nSaved: {report_path}\nSaved: {csv_path}")


def cmd_convert(args, _paths: Paths):
    from rag_ingest.convert.cli import run
    sys.exit(run(args))


def cmd_ingest(args, paths: Paths):
    from rag_ingest.ingest import DEFAULT_SOURCES, run_ingest
    from rag_ingest.stores import open_store

    stores = [open_store(paths, d) for d in (args.ocr_dir or DEFAULT_SOURCES)]
    result = run_ingest(paths.resolve(args.docs_dir), paths.resolve(args.output_dir),
                        stores, args.min_side)
    for e in result.manifest:
        if e.get("rejected") and "duplicate_of" not in e:
            reasons = ", ".join(f"{r['source']} ({r['reason']})" for r in e["rejected"])
            print(f"Rejected OCR for {e['image']} from {reasons} -> {e.get('source', 'none')}")
    print(f"\nImages: {len(result.manifest)} | icons skipped: {result.count('icon')} | "
        f"duplicates: {result.count('duplicate')} | unique to OCR: {len(result.unique)} "
        f"| still need OCR: {len(result.needs_ocr)}"
        + (f" | unsupported format: {n}" if (n := result.count("unsupported_format")) else ""))
    print(f"Results in: {paths.resolve(args.output_dir)}")


OUTPUT_DIR_FLAG = "--output-dir"


def add_docs_dir(parser, paths: Paths) -> None:
    parser.add_argument("--docs-dir", "--pdf-dir", dest="docs_dir",
                        default=paths.relative(paths.documents),
                        help="Folder of .pdf / .docx files (--pdf-dir is the old name).")


def build_parser(paths: Paths) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag-ingest", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    rel = lambda p: paths.relative(p)

    def add(name, func, help_):
        p = sub.add_parser(name, help=help_, description=help_)
        p.set_defaults(func=func)
        return p

    p = add("inspect", cmd_inspect, "Text characters and image count per document page.")
    add_docs_dir(p, paths)

    p = add("extract-images", cmd_extract_images, "Save every embedded image.")
    add_docs_dir(p, paths)
    p.add_argument(OUTPUT_DIR_FLAG, default=rel(paths.images))

    p = add("extract-text", cmd_extract_text, "Save each document's plain text, page-marked.")
    add_docs_dir(p, paths)
    p.add_argument(OUTPUT_DIR_FLAG, default=rel(paths.outputs))

    p = add("ocr", cmd_ocr, "Run one OCR engine on one image or a folder.")
    p.add_argument("--engine", choices=["tesseract", "paddle", "ollama", "api"], required=True)
    target = p.add_mutually_exclusive_group()
    target.add_argument("--image", help="One image instead of the whole folder.")
    target.add_argument("--images-dir", default=rel(paths.images))
    p.add_argument(OUTPUT_DIR_FLAG, help="Default: the engine's folder "
                "(tesseract, ocr, or vision for ollama/api).")
    p.add_argument("--skip-existing", action="store_true", help="Resume: skip images with a result.")
    p.add_argument("--upscale", type=int, default=0, metavar="PX",
                help="Enlarge images whose shorter side is below PX (max 4x).")
    p.add_argument("--psm", type=int, default=3, help="Tesseract page segmentation mode.")
    p.add_argument("--hint", nargs="?", const="paddle", choices=["paddle", "tesseract", "both"],
                help="Vision engines: add confident line-OCR results to the prompt "
                        "(default source: paddle).")
    p.add_argument("--prompt", metavar="FILE",
                   help="Vision engines: prompt file (default prompts/thai_document_ocr.txt). "
                        "Also: prompts/thai_document_ocr_compact.txt, or _full.txt (9k tokens, "
                        "too long for qwen2.5vl:3b's 8k context). Recorded in each result.")
    p.add_argument("--model", help="Ollama model (default qwen2.5vl:3b).")
    p.add_argument("--num-ctx", type=int, default=8192, help="Ollama context size.")
    p.add_argument("--api-url", default=os.environ.get("OCR_API_URL"),
                help="Full /v1/chat/completions URL (or env OCR_API_URL).")
    p.add_argument("--api-model", default=os.environ.get("OCR_API_MODEL"),
                help="Model name for the API (or env OCR_API_MODEL).")

    p = add("review", cmd_review, "Correct OCR results in a local web page.")
    p.add_argument("--dir", default="outputs/vision")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true")

    p = add("compare", cmd_compare, "Compare two vision result folders.")
    p.add_argument("--a", default="outputs/vision")
    p.add_argument("--b", default="outputs/vision_hint")
    p.add_argument("--diff", action="store_true", help="Add a line diff per differing image.")
    p.add_argument("--report", default="outputs/compare_report.md", help="'' to skip saving.")

    p = add("benchmark", cmd_benchmark, "Score every method against corrected text (CER).")
    p.add_argument("--ground-truth-dir", default=rel(paths.ground_truth))
    p.add_argument("--report", default="outputs/benchmark_report.md")
    p.add_argument("--csv", default="outputs/benchmark.csv")
    p.add_argument("--exclude", nargs="*", default=[], metavar="METHOD")

    p = add("ingest", cmd_ingest, "Build RAG text per document (PDF/Word): its text + screenshot OCR.")
    add_docs_dir(p, paths)
    p.add_argument("--ocr-dir", action="append",
                help="OCR result folder; repeat for fallbacks, best first "
                        "(default: vision_typhoon, then vision_hint).")
    p.add_argument("--min-side", type=int, default=120,
                help="Images whose shorter side is below this are icons (skipped).")
    p.add_argument(OUTPUT_DIR_FLAG, default=rel(paths.ingest))

    from rag_ingest.convert.cli import USAGE_EXAMPLES, add_arguments as add_convert_arguments
    p = add("convert", cmd_convert, "Convert PDF to Word (.docx) keeping layout; Thai-aware.")
    p.epilog, p.formatter_class = USAGE_EXAMPLES, argparse.RawDescriptionHelpFormatter
    add_convert_arguments(p, paths)
    return parser


def main(argv: list[str] | None = None) -> None:
    sys.stdout.reconfigure(encoding=ENCODING)  # Thai garbles on the default Windows console
    paths = Paths.default()
    args = build_parser(paths).parse_args(argv)
    args.func(args, paths)


if __name__ == "__main__":
    main()
