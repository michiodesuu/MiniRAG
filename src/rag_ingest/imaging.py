"""Image preparation shared by the OCR engines."""
from pathlib import Path
import io

from PIL import Image

MAX_UPSCALE = 4
# Tiling: a strip may end up to this share of the strip height above or below
# its target, so the cut can land on a blank row between lines of text.
CUT_WINDOW = 0.25


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


def png_bytes(img: Image.Image) -> bytes:
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def cut_rows(img: Image.Image, strip_height: int) -> list[int]:
    """Rows at which to cut img into strips of about strip_height: near each
    target, the quietest row (least variation in brightness), so a cut runs
    through the gap between lines of text rather than through a line."""
    import numpy as np

    if not strip_height or img.height <= strip_height * (1 + CUT_WINDOW):
        return []  # one strip; also avoids a sliver at the bottom
    spread = np.asarray(img.convert("L"), dtype=np.float32).std(axis=1)
    window = round(strip_height * CUT_WINDOW)
    cuts, top = [], 0
    while img.height - top > strip_height * (1 + CUT_WINDOW):
        target = top + strip_height
        low, high = target - window, min(target + window, img.height - 1)
        cuts.append(low + int(np.argmin(spread[low:high + 1])))
        top = cuts[-1]
    return cuts


def strips(img: Image.Image, strip_height: int) -> list[Image.Image]:
    """img cut into horizontal strips (see cut_rows); [img] when it is short enough."""
    edges = [0, *cut_rows(img, strip_height), img.height]
    return [img.crop((0, top, img.width, bottom)) for top, bottom in zip(edges, edges[1:])]


def list_images(folder: Path, extensions: frozenset[str]) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in extensions)
