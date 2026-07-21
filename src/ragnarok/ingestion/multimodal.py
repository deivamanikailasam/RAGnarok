"""Multimodal ingest support (Step 36).

Images/diagrams are turned into retrievable text at ingest by *captioning* them (a local
vision-language model in prod: LLaVA / Qwen-VL / a caption model), then embedding the caption
alongside the alt text. The chunk keeps a reference to the asset so the generator can cite/link it.
Captioner is a pluggable hook so the package works with no VLM installed (default: use the alt text).
"""

from __future__ import annotations

from collections.abc import Callable

# fn(src, alt) -> caption. Default returns the alt text; prod plugs a VLM.
Captioner = Callable[[str, str], str]
_captioner: Captioner | None = None


def set_image_captioner(fn: Captioner | None) -> None:
    global _captioner
    _captioner = fn


def caption_image(src: str, alt: str) -> str:
    if _captioner is not None:
        try:
            return _captioner(src, alt)
        except Exception:  # noqa: BLE001 - captioning must never break ingest
            pass
    return alt or f"Image asset at {src}"
