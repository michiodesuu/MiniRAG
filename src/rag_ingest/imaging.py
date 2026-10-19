"""Image preparation shared by the OCR engines."""
from pathlib import Path
import io

from PIL import Image

MAX_UPSCALE = 4


def upscaled(img: Image.Image, min_side: int) -> Image.Image:
    """Enlarge so the shorter side reaches min_side px (at most 4x). Never shrinks."""
    scale = min_side / min(img.size) if min_side else 1
    if scale <= 1:
        return img
    scale = min(scale, MAX_UPSCALE)
    return img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)


def load_rgb(path: Path, min_side: int = 0) -> Image.Image:
    with Image.open(path) as img:
        return upscaled(img.convert("RGB"), min_side)


# Formats vision models and image APIs reliably accept as-is.
SENDABLE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg"})


def image_bytes(path: Path, min_side: int = 0) -> bytes:
    """The original file's bytes, or a PNG re-encode if it must be upscaled or is
    in a format vision models may reject (GIF, BMP, TIFF, WebP)."""
    with Image.open(path) as img:
        needs_upscale = bool(min_side) and min(img.size) < min_side
        if needs_upscale or path.suffix.lower() not in SENDABLE_SUFFIXES:
            buffer = io.BytesIO()
            upscaled(img.convert("RGB"), min_side).save(buffer, format="PNG")
            return buffer.getvalue()
    return path.read_bytes()


def list_images(folder: Path, extensions: frozenset[str]) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in extensions)
