"""Unified media input handling.

Input forms: local path / file:// URL / http(s) URL / base64 data URI.

Resolution rules (per official MiMo docs):
- image: URL passes through (cloud scales); local/base64 is format-checked, hard-capped
  to 8MP and (by default) compressed for OCR/token savings, then sent as a data URI.
- video: URL passes through; local files are re-encoded via ffmpeg to target
  resolution + fps, then sent as an mp4 data URI (≤50MB).
- audio: URL is passed through directly in ``input_audio.data`` (per docs);
  local/base64 becomes a ``data:{mime};base64,`` URI.
"""

import base64
import binascii
import os
import re
from io import BytesIO
from typing import Optional
from urllib.parse import unquote, urlparse

from .. import config
from . import image_proc
from . import video_proc


class InputError(Exception):
    """Raised when an image/video input cannot be read or fails validation."""


_DATA_URI_RE = re.compile(r"^data:(?P<mime>[\w./+-]+)?;base64,(?P<data>.+)$", re.DOTALL)

# 白名单(官方文档) -> MIME
_VIDEO_EXT_MIME = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".wmv": "video/x-ms-wmv",
}


# --- 兼容旧引用:返回硬编码 config 值 ---
def _max_size() -> int:
    return config.MAX_IMAGE_SIZE


def _max_video_size() -> int:
    return config.MAX_VIDEO_SIZE


def _max_audio_size() -> int:
    return config.MAX_AUDIO_SIZE


def _allowed_formats() -> set[str]:
    return set(config.IMAGE_FORMATS)


# --- 基础工具 ---
def is_http_url(src: str) -> bool:
    return src.startswith("http://") or src.startswith("https://")


def is_data_uri(src: str) -> bool:
    return src.startswith("data:")


def looks_like_base64(src: str) -> bool:
    if len(src) < 32:
        return False
    return re.fullmatch(r"[A-Za-z0-9+/=\s]+", src) is not None


def _normalize_path(src: str) -> str:
    if src.startswith("file://"):
        parsed = urlparse(src)
        path = unquote(parsed.path)
    else:
        path = src
    if "\x00" in path:
        raise InputError("path contains a null byte")
    return os.path.abspath(os.path.expanduser(path))


def existing_path(src: str) -> Optional[str]:
    """Return a normalized path if src refers to an existing file, else None."""
    if src.startswith("data:"):
        return None
    try:
        path = _normalize_path(src)
    except InputError:
        return None
    return path if os.path.isfile(path) else None


def local_path(src: str) -> str:
    """Normalize a path / file:// URL to a validated local file path."""
    path = _normalize_path(src)
    if not os.path.isfile(path):
        raise InputError(f"file not found: {path}")
    return path


def check_size(data: bytes) -> None:
    if len(data) > config.MAX_IMAGE_SIZE:
        raise InputError(f"image exceeds {config.MAX_IMAGE_SIZE} bytes")


def sniff_image_format(data: bytes) -> str:
    """Return a lowercase image format (e.g. 'jpeg') using Pillow; validates it is an image."""
    from PIL import Image

    try:
        with Image.open(BytesIO(data)) as img:
            fmt = (img.format or "").lower()
    except Exception as e:
        raise InputError(f"not a readable image: {e}")
    if fmt == "jpg":
        fmt = "jpeg"
    if not fmt:
        raise InputError("could not determine image format")
    return fmt


def check_allowed_format(fmt: str) -> None:
    if fmt not in config.IMAGE_FORMATS:
        raise InputError(f"image format '{fmt}' not allowed (allowed: {sorted(config.IMAGE_FORMATS)})")


def decode_base64(src: str) -> bytes:
    payload = src
    m = _DATA_URI_RE.match(src)
    if m:
        payload = m.group("data")
    try:
        return base64.b64decode(payload, validate=False)
    except (binascii.Error, ValueError) as e:
        raise InputError(f"invalid base64 data: {e}")


def _data_uri(mime: str, data: bytes) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


async def load_image_bytes(src: str) -> tuple[bytes, str]:
    """Load a non-URL image input into (bytes, format), validating size and format."""
    src = src.strip()
    if is_data_uri(src):
        data = decode_base64(src)
    else:
        path = existing_path(src)
        if path:
            with open(path, "rb") as f:
                data = f.read()
        elif looks_like_base64(src):
            data = decode_base64(src)
        else:
            raise InputError(f"file not found: {_normalize_path(src)}")
    check_size(data)
    fmt = sniff_image_format(data)
    check_allowed_format(fmt)
    return data, fmt


# --- 图片 ---
async def resolve_image_url(src: str, *, compress: bool | None = None) -> str:
    """Return value for ``image_url.url``.

    http(s) URL passes through (cloud fetches + scales); local/base64 is capped to
    8MP and (default) compressed, then returned as a data URI.
    """
    src = src.strip()
    if is_http_url(src):
        return src
    if compress is None:
        compress = config.IMAGE_COMPRESS
    data, fmt = await load_image_bytes(src)
    if compress:
        data2, fmt2 = image_proc.process_image(data)
        return _data_uri(f"image/{fmt2}", data2)
    # 不压缩:仅 8MP 硬上限
    data2, fmt2 = image_proc.process_image_without_compress(data)
    return _data_uri(f"image/{fmt2}", data2)


# --- 视频 ---
async def resolve_video(src: str, *, fps: float | None = None,
                        media_resolution: str | None = None) -> dict:
    """Return the ``video_url`` object.

    URL -> pass through (optionally with fps / media_resolution per docs).
    Local -> re-encode via ffmpeg (resolution + fps), return mp4 data URI.
    """
    src = src.strip()
    if is_http_url(src):
        obj = {"url": src}
        if fps is not None:
            obj["fps"] = fps
        if media_resolution:
            obj["media_resolution"] = media_resolution
        return obj
    if is_data_uri(src):
        return {"url": src}

    path = existing_path(src)
    if path:
        ext = os.path.splitext(path.lower())[1]
        if ext not in _VIDEO_EXT_MIME:
            raise InputError(
                f"video format '{ext.lstrip('.')}' not allowed (allowed: {sorted(config.VIDEO_FORMATS)})"
            )
        out = video_proc.reencode_video(path, fps=fps)
        try:
            with open(out, "rb") as f:
                data = f.read()
        finally:
            try:
                os.unlink(out)
            except OSError:
                pass
        if len(data) > config.MAX_VIDEO_SIZE:
            raise InputError(
                f"re-encoded video exceeds {config.MAX_VIDEO_SIZE} bytes; "
                "reduce duration/resolution or pass an http(s) URL"
            )
        return {"url": _data_uri("video/mp4", data)}

    if looks_like_base64(src):
        return {"url": f"data:video/mp4;base64,{src}"}

    raise InputError(f"file not found: {_normalize_path(src)}")


# --- 音频 ---
_AUDIO_FMT_MIME = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "flac": "audio/flac",
    "m4a": "audio/mp4",
    "ogg": "audio/ogg",
}
_AUDIO_MIME_FMT = {mime: fmt for fmt, mime in _AUDIO_FMT_MIME.items()}


async def resolve_audio(src: str, *, max_size: int, formats: set[str] | None = None) -> tuple[str, str | None]:
    """Return (value, format) for ``input_audio``.

    http(s) URL -> (url, None): 按官方文档直放 data 字段(注意:opencode 网关收不到 URL 音频)。
    Local / base64 -> (bare_b64, fmt): 走 OpenAI 标准 ``data`` + ``format``。

    ``formats`` optionally restricts allowed formats (e.g. ASR: wav/mp3 only).
    """
    src = src.strip()
    if is_http_url(src):
        return src, None
    if is_data_uri(src):
        data = decode_base64(src)
        m = re.match(r"^data:([\w./+-]+)", src, re.IGNORECASE)
        fmt = _AUDIO_MIME_FMT.get((m.group(1) if m else "").lower())
        if not fmt:
            raise InputError("could not determine audio format from data URI")
    else:
        path = existing_path(src)
        if not path:
            raise InputError(f"file not found: {_normalize_path(src)}")
        fmt = os.path.splitext(path.lower())[1].lstrip(".")
        with open(path, "rb") as f:
            data = f.read()
    if fmt not in config.AUDIO_FORMATS:
        raise InputError(f"audio format '{fmt}' not allowed (allowed: {sorted(config.AUDIO_FORMATS)})")
    if formats and fmt not in formats:
        raise InputError(f"audio format '{fmt}' not allowed here (allowed: {sorted(formats)})")
    if len(data) > max_size:
        raise InputError(f"audio exceeds size limit ({len(data)} > {max_size} bytes)")
    return base64.b64encode(data).decode("ascii"), fmt
