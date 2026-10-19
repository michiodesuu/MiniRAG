"""Vision-language OCR: one engine, interchangeable transports (Bridge).

VisionEngine owns what is sent (prompt, hints, image preparation);
a Transport owns how it is sent (local Ollama or an OpenAI-compatible API).
"""
from pathlib import Path
from typing import Protocol
import base64

from rag_ingest.config import ENCODING, HINT_FOLDERS, Paths
from rag_ingest.imaging import image_bytes
from rag_ingest.models import Recognition
from rag_ingest.stores import LineStore

TEMPERATURE = 0
HINT_MIN_CONFIDENCE = 0.8
HINT_INTRO = ("\n\nREFERENCE (separate OCR engines read these lines; they may be "
              "wrong or incomplete, so trust the image over them):\n")


class Transport(Protocol):
    model: str

    def complete(self, prompt: str, image: bytes) -> str: ...

    def settings(self) -> dict: ...


class OllamaTransport:
    def __init__(self, model: str = "qwen2.5vl:3b", num_ctx: int = 8192):
        import ollama

        self._ollama = ollama
        # Ollama's default context (4096) is too small once images are upscaled.
        self.model, self.num_ctx = model, num_ctx

    def complete(self, prompt: str, image: bytes) -> str:
        response = self._ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt, "images": [image]}],
            options={"temperature": TEMPERATURE, "num_ctx": self.num_ctx},
        )
        return response["message"]["content"].strip()

    def settings(self) -> dict:
        return {"backend": "ollama", "num_ctx": self.num_ctx}


class OpenAICompatibleTransport:
    """POST /v1/chat/completions with a base64 image (e.g. the Typhoon OCR API)."""

    def __init__(self, url: str, model: str, key: str | None = None, timeout: float = 300):
        import httpx

        self._httpx = httpx
        self.url, self.model, self.key, self.timeout = url, model, key, timeout

    def complete(self, prompt: str, image: bytes) -> str:
        mime = "image/png" if image[:4] == b"\x89PNG" else "image/jpeg"
        data_url = f"data:{mime};base64,{base64.b64encode(image).decode()}"
        response = self._httpx.post(
            self.url,
            headers={"Authorization": f"Bearer {self.key}"} if self.key else {},
            timeout=self.timeout,
            json={
                "model": self.model,
                "temperature": TEMPERATURE,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ]}],
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

    def settings(self) -> dict:
        return {"backend": "api"}


class HintProvider:
    """Confident lines from line-OCR results, added to the prompt as a reference."""

    def __init__(self, stores: dict[str, LineStore], min_confidence: float = HINT_MIN_CONFIDENCE):
        self.stores, self.min_confidence = stores, min_confidence

    def hint(self, stem: str) -> tuple[str | None, list[str]]:
        """(hint text or None, names of the sources that contributed)."""
        parts = [(name, "\n".join(lines)) for name, store in self.stores.items()
                 if (lines := store.confident_lines(stem, self.min_confidence))]
        if not parts:
            return None, []
        used = [name for name, _ in parts]
        if len(parts) == 1:
            return parts[0][1], used
        return "\n\n".join(f"[{name} OCR]\n{text}" for name, text in parts), used


class VisionEngine:
    store_kind = "vision"
    default_output = "vision"

    def __init__(self, transport: Transport, prompt: str,
                 hints: HintProvider | None = None, upscale: int = 0,
                 prompt_file: Path | None = None):
        self.transport, self.prompt, self.hints, self.upscale = transport, prompt, hints, upscale
        self.prompt_file = prompt_file

    def recognize(self, image: Path) -> Recognition:
        hint, used = self.hints.hint(image.stem) if self.hints else (None, [])
        prompt = self.prompt + (HINT_INTRO + hint if hint else "")
        text = self.transport.complete(prompt, image_bytes(image, self.upscale))
        return Recognition(
            text=text,
            model=self.transport.model,
            settings={
                **self.transport.settings(),
                "temperature": TEMPERATURE,
                "paddle_hint": "paddle" in used,
                "tesseract_hint": "tesseract" in used,
                "upscale_min_side": self.upscale,
            },
            prompt_file=self.prompt_file,
        )


def _hint_provider(paths: Paths, hint_source: str | None) -> HintProvider | None:
    if not hint_source:
        return None
    names = list(HINT_FOLDERS) if hint_source == "both" else [hint_source]
    return HintProvider({n: LineStore(paths.method(HINT_FOLDERS[n])) for n in names})


def _vision_engine(transport: Transport, paths: Paths, hint_source: str | None,
                   upscale: int, prompt: Path | None) -> VisionEngine:
    prompt_file = prompt or paths.prompt
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt not found: {prompt_file}")
    return VisionEngine(transport, prompt_file.read_text(encoding=ENCODING),
                        _hint_provider(paths, hint_source), upscale, prompt_file)


def ollama_engine(paths: Paths, hint_source: str | None = None, upscale: int = 0,
                  model: str = "qwen2.5vl:3b", num_ctx: int = 8192,
                  prompt: Path | None = None) -> VisionEngine:
    return _vision_engine(OllamaTransport(model, num_ctx), paths, hint_source, upscale, prompt)


def api_engine(paths: Paths, url: str, model: str, key: str | None = None,
               hint_source: str | None = None, upscale: int = 0,
               prompt: Path | None = None) -> VisionEngine:
    return _vision_engine(OpenAICompatibleTransport(url, model, key), paths, hint_source,
                          upscale, prompt)
