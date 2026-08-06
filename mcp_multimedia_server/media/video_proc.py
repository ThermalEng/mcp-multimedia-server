"""Video re-encode via ffmpeg: target resolution + fps.

Local videos are re-encoded before sending so the payload stays small and the
resolution / frame rate match the configured (or requested) precision. The short
edge is capped at 720 (720p), the long edge scales proportionally.
"""

import os
import subprocess
import tempfile

from .. import config


def reencode_video(src_path: str, *, fps: float | None = None) -> str:
    """Re-encode a local video to an mp4 at 720p short edge + target fps. Returns output path."""
    from .inputs import InputError  # lazy: avoid circular import at module load

    target_fps = fps if fps is not None else config.VIDEO_TARGET_FPS
    if not (1 <= target_fps <= 30):
        raise InputError(f"fps out of range [1, 30]: {target_fps}")

    fd, out = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    se = config.VIDEO_TARGET_SHORT_EDGE
    # 短边缩到 ≤ se(720p),长边等比;强制偶尺寸(libx264/yuv420p 要求)
    scale = (
        "scale="
        f"w='trunc(iw*min({se},min(iw,ih))/min(iw,ih)/2)*2':"
        f"h='trunc(ih*min({se},min(iw,ih))/min(iw,ih)/2)*2'"
    )
    cmd = [
        config.VIDEO_FFMPEG, "-loglevel", "error", "-y", "-i", src_path,
        "-vf", scale,
        "-r", str(target_fps),
        "-c:v", "libx264", "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        "-map", "0:v:0", "-map", "0:a?",      # 音轨可选(源无音频时忽略)
        "-c:a", "aac", "-b:a", "96k",         # 保留音轨(官方视频理解含音频 token)
        out,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except FileNotFoundError:
        raise InputError("ffmpeg not found; set config.VIDEO_FFMPEG to a valid path")
    except subprocess.CalledProcessError as e:
        detail = (e.stderr or b"").decode(errors="replace")[-300:]
        try:
            os.unlink(out)
        except OSError:
            pass
        raise InputError(f"ffmpeg re-encode failed: {detail}")
    return out
