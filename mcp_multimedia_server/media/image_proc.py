"""Image pre-processing: 8MP hard cap + optional compress for OCR / token savings.

All local/base64 images go through here before being sent, per official MiMo docs
(image is auto-scaled to ≤8.4MP server-side anyway; client-side downscale first
saves upload size, and going below that saves tokens since image tokens scale with
resolution²).
"""

from io import BytesIO

from PIL import Image

from .. import config


def process_image(data: bytes) -> tuple[bytes, str]:
    """Return (bytes, format).

    - Always: downscale to ≤ config.IMAGE_MAX_PIXELS (8MP).
    - If config.IMAGE_COMPRESS: also downscale to target long edge and re-encode as
      JPEG (quality config.IMAGE_JPEG_QUALITY).
    - If not compress: keep original format.
    """
    img = Image.open(BytesIO(data))
    img.load()
    w, h = img.size

    scale = 1.0
    if w * h > config.IMAGE_MAX_PIXELS:
        scale = min(scale, (config.IMAGE_MAX_PIXELS / (w * h)) ** 0.5)
    if config.IMAGE_COMPRESS:
        long_edge = max(w, h)
        if long_edge > config.IMAGE_TARGET_LONG_EDGE:
            scale = min(scale, config.IMAGE_TARGET_LONG_EDGE / long_edge)
    if scale < 1.0:
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)

    buf = BytesIO()
    if config.IMAGE_COMPRESS:
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        img.save(buf, format="JPEG", quality=config.IMAGE_JPEG_QUALITY)
        return buf.getvalue(), "jpeg"

    fmt = (img.format or "jpeg").lower()
    if fmt == "jpg":
        fmt = "jpeg"
    img.save(buf, format=fmt)
    return buf.getvalue(), fmt


def process_image_without_compress(data: bytes) -> tuple[bytes, str]:
    """Only the 8MP hard cap; keeps original format and quality (no re-encode)."""
    img = Image.open(BytesIO(data))
    img.load()
    w, h = img.size
    if w * h > config.IMAGE_MAX_PIXELS:
        scale = (config.IMAGE_MAX_PIXELS / (w * h)) ** 0.5
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    buf = BytesIO()
    fmt = (img.format or "jpeg").lower()
    if fmt == "jpg":
        fmt = "jpeg"
    img.save(buf, format=fmt)
    return buf.getvalue(), fmt
