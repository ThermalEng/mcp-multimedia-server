#!/usr/bin/env python3
"""MCP Multimedia Server - cloud image/video/audio analysis tools for LLM agents.

All media understanding is delegated to a cloud OpenAI-compatible multimodal model.
- images  -> ``image_url`` (local images compressed: 8MP cap + OCR target)
- videos  -> ``video_url`` (local files ffmpeg re-encoded to low res + fps)
- audio   -> ``input_audio`` (URL passed through per docs; local/base64 as data URI)

Operational limits/params are hardcoded in ``config.py``; the upstream trio
(BASE_URL / API_KEY / MODEL) comes from process env (Claude Code MCP registration).
"""

import asyncio
import hashlib
import json
import os
import sys
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.server.sse import SseServerTransport
from mcp.types import Tool, TextContent

from . import config
from .providers.base import ProviderError, VisionResult
from .providers.openai_compat import OpenAICompatProvider
from .media import inputs, imageinfo
from .media.inputs import InputError
from .media.imageinfo import ImageInfoError

server = Server("mcp-multimedia-server")

_provider = None

PROMPT_PRESETS = {
    "describe": "请详细描述这张图片的内容，包括主要对象、场景、颜色以及所有可见细节。",
    "ocr": "请提取图片中所有可见的文字，尽量保持原始布局与顺序，不要翻译或改写。",
    "chart": "这是一张图表。请判断其类型，说明坐标轴/图例含义，并总结关键数据与趋势。",
    "ui": "这是一张 UI 界面截图。请分析其布局结构、主要组件、配色方案与交互元素。",
    "diagram": "这是一张示意图或流程图。请解释其中的节点、连接关系与整体流程。",
    "error": "这是一张报错截图。请识别其中的错误信息，分析可能的原因，并给出修复建议。",
}


def _init_provider():
    global _provider
    _provider = OpenAICompatProvider(
        base_url=config.BASE_URL,
        api_key=config.API_KEY,
        model=config.MODEL,
        timeout=config.TIMEOUT,
        max_retries=config.MAX_RETRIES,
        default_max_tokens=config.MAX_TOKENS,
        default_temperature=config.TEMPERATURE,
        reasoning_effort=config.REASONING_EFFORT,
    )


def _get_provider() -> OpenAICompatProvider:
    if _provider is None:
        _init_provider()
    if not _provider.is_available():
        raise ProviderError(
            "openai",
            "Provider not configured. Set MCP_MEDIA_BASE_URL, MCP_MEDIA_API_KEY and MCP_MEDIA_MODEL.",
        )
    return _provider


_IMG_MAX_MB = config.MAX_IMAGE_SIZE // (1024 * 1024)
_VID_MAX_MB = config.MAX_VIDEO_SIZE // (1024 * 1024)
_AUD_MAX_MB = config.MAX_AUDIO_SIZE // (1024 * 1024)
_ASR_MAX_MB = config.MAX_ASR_SIZE // (1024 * 1024)

_IMAGE_PROP = {
    "type": "string",
    "description": (
        "图片，支持：本地文件路径 / http(s) URL / base64(data URI)。"
        f"单张最大 {_IMG_MAX_MB}MB，格式 jpeg/png/gif/webp/bmp"
    ),
}
_IMAGE_BATCH_PROP = {
    "type": "array",
    "items": {"type": "string"},
    "description": f"多张图片数组（如文档逐页扫描）：每项同 image；一次最多 {config.MAX_IMAGES_PER_BATCH} 张",
}
_VIDEO_PROP = {
    "type": "string",
    "description": (
        "视频，支持：本地文件路径 / http(s) URL / base64(data URI)。"
        f"本地视频会自动压缩；最大 {_VID_MAX_MB}MB，格式 mp4/mov/avi/wmv"
    ),
}
_AUDIO_PROP = {
    "type": "string",
    "description": (
        "音频，支持：本地文件路径 / http(s) URL / base64(data URI)。"
        f"推荐用本地文件或 base64（URL 音频可能不被识别）。最大 {_AUD_MAX_MB}MB，格式 mp3/wav/flac/m4a/ogg"
    ),
}
_ASR_AUDIO_PROP = {
    "type": "string",
    "description": (
        "音频，支持：本地文件路径 / base64(data URI)。"
        f"仅 wav/mp3，最大 {_ASR_MAX_MB}MB"
    ),
}
_LANGUAGE_PROP = {
    "type": "string",
    "enum": ["auto", "zh", "en"],
    "description": "转写语种：auto 自动检测 / zh 中文 / en 英文（默认 auto，明确语种识别更准）",
}


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="analyze_image",
            description=(
                "图片理解/OCR：分析一张或多张图片，支持描述、文字识别(OCR)、图表/UI/报错诊断等。"
                "单张用 image，多张（如文档多页）用 image_batch 数组。"
                "可用 preset 选任务类型(describe/ocr/chart/ui/diagram/error)或自由写 prompt。"
                "本地图片自动压缩优化（OCR 足够清晰、更快更省），需原图细节可设 compress=false。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "image": _IMAGE_PROP,
                    "image_batch": _IMAGE_BATCH_PROP,
                    "prompt": {"type": "string", "description": "自由指令/问题；提供后覆盖 preset", "default": ""},
                    "preset": {
                        "type": "string",
                        "enum": list(PROMPT_PRESETS.keys()),
                        "description": "任务预设：describe 描述 / ocr 文字识别 / chart 图表 / ui 界面 / diagram 示意图 / error 报错诊断",
                        "default": "describe",
                    },
                    "compress": {"type": "boolean", "description": "是否压缩本地图片（默认开；关掉保留原图细节，但更耗 token）"},
                    "max_tokens": {"type": "integer", "description": f"可选，本次生成上限（默认 {config.MAX_TOKENS}）"},
                    "temperature": {"type": "number", "description": "可选，采样温度"},
                },
            },
        ),
        Tool(
            name="analyze_video",
            description=(
                "视频理解：让模型描述视频内容、按时间顺序总结。支持本地文件、URL 或 base64，本地视频自动压缩。"
                "fps 控制抽帧密度（默认 2，范围 1-30；越大时序越精细、越耗 token），一般用默认即可。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "video": _VIDEO_PROP,
                    "prompt": {"type": "string", "description": "自由指令/问题，如'这段视频里发生了什么？按时间顺序描述'", "default": ""},
                    "fps": {"type": "number", "description": "抽帧密度，范围 [1,30]，默认 2（越大时序越精细、越耗 token）"},
                    "media_resolution": {
                        "type": "string",
                        "enum": ["default", "max"],
                        "description": "分辨率档次：default 平衡 / max 细节增强（仅 URL 视频可选）",
                    },
                    "max_tokens": {"type": "integer", "description": f"可选，本次生成上限（默认 {config.MAX_TOKENS}）"},
                    "temperature": {"type": "number", "description": "可选，采样温度"},
                },
                "required": ["video"],
            },
        ),
        Tool(
            name="analyze_audio",
            description=(
                "音频理解/STT：转录或总结一段音频内容。推荐用本地文件或 base64（URL 音频可能不被识别）。"
                "不传 prompt 默认转录并总结主旨。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "audio": _AUDIO_PROP,
                    "prompt": {"type": "string", "description": "自由指令/问题，如'转录这段语音'；不传则默认转录并总结主旨", "default": ""},
                    "max_tokens": {"type": "integer", "description": f"可选，本次生成上限（默认 {config.MAX_TOKENS}）"},
                    "temperature": {"type": "number", "description": "可选，采样温度"},
                },
                "required": ["audio"],
            },
        ),
        Tool(
            name="asr",
            description=(
                "语音转写(ASR)：把音频转成纯文本，适合会议记录、方言、嘈杂环境录音。"
                "仅支持 wav/mp3，最大 10MB。用 language 明确语种(auto/zh/en)可提高准确率。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "audio": _ASR_AUDIO_PROP,
                    "language": _LANGUAGE_PROP,
                    "max_tokens": {"type": "integer", "description": f"可选，本次生成上限（默认 {config.MAX_TOKENS}）"},
                },
                "required": ["audio"],
            },
        ),
        Tool(
            name="image_metadata",
            description="读取本地图片的元信息（尺寸、格式、颜色模式、EXIF/GPS）。需本地文件路径，不调用云端。",
            inputSchema={
                "type": "object",
                "properties": {"image": _IMAGE_PROP},
                "required": ["image"],
            },
        ),
        Tool(
            name="get_server_status",
            description="查询服务状态：云端模型是否已配置、当前模型与各项限制/处理参数。",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    try:
        result = await _handle_tool(name, arguments)
        return [TextContent(type="text", text=result)]
    except ProviderError as e:
        return [TextContent(type="text", text=json.dumps({"error": str(e), "provider": e.provider}, ensure_ascii=False))]
    except (InputError, ImageInfoError) as e:
        return [TextContent(type="text", text=json.dumps({"error": str(e)}, ensure_ascii=False))]
    except Exception as e:
        return [TextContent(type="text", text=json.dumps({"error": str(e)}, ensure_ascii=False))]


async def _handle_tool(name: str, args: dict[str, Any]) -> str:
    if name == "analyze_image":
        return await _analyze_image(args)
    elif name == "analyze_video":
        return await _analyze_video(args)
    elif name == "analyze_audio":
        return await _analyze_audio(args)
    elif name == "asr":
        return await _analyze_asr(args)
    elif name == "image_metadata":
        return await _image_metadata(args)
    elif name == "get_server_status":
        return _server_status()
    else:
        return json.dumps({"error": f"Unknown tool: {name}"}, ensure_ascii=False)


def _optional(args: dict, key: str):
    value = args.get(key)
    return value if value not in (None, "") else None


async def _analyze_image(args: dict) -> str:
    provider = _get_provider()
    preset = args.get("preset", "describe")
    prompt = args.get("prompt") or PROMPT_PRESETS.get(preset, PROMPT_PRESETS["describe"])
    compress = args.get("compress")
    compress = None if compress is None else bool(compress)
    images = _collect_images(args)
    if not images:
        raise InputError("provide 'image' (single) or 'images' (array)")
    if len(images) > config.MAX_IMAGES_PER_BATCH:
        raise InputError(
            f"too many images ({len(images)} > {config.MAX_IMAGES_PER_BATCH}); reduce count or split into batches"
        )
    parts = []
    for src in images:
        url = await inputs.resolve_image_url(src, compress=compress)
        parts.append({"type": "image_url", "image_url": {"url": url}})
    parts.append({"type": "text", "text": prompt})
    result = await _cached_analyze(provider, parts, args)
    out = result.to_dict()
    out["preset"] = preset if not args.get("prompt") else None
    out["image_count"] = len(images)
    return json.dumps(out, ensure_ascii=False)


def _collect_images(args: dict) -> list[str]:
    many = args.get("image_batch")
    if many:
        if isinstance(many, str):
            many = [many]
        return [str(x) for x in many]
    single = args.get("image")
    return [str(single)] if single else []


async def _analyze_video(args: dict) -> str:
    provider = _get_provider()
    prompt = args.get("prompt") or (
        "请按时间顺序详细描述这个视频的内容：场景、主要对象、发生的动作与事件。"
    )
    fps = _optional(args, "fps")
    fps = float(fps) if fps is not None else None
    media_resolution = args.get("media_resolution") or None
    obj = await inputs.resolve_video(args["video"], fps=fps, media_resolution=media_resolution)
    parts = [
        {"type": "video_url", "video_url": obj},
        {"type": "text", "text": prompt},
    ]
    result = await _cached_analyze(provider, parts, args)
    return json.dumps(result.to_dict(), ensure_ascii=False)


async def _analyze_audio(args: dict) -> str:
    provider = _get_provider()
    prompt = args.get("prompt") or "请转录这段语音的内容，并简要总结其主旨。"
    audio_val, audio_fmt = await inputs.resolve_audio(args["audio"], max_size=config.MAX_AUDIO_SIZE)
    aa = {"data": audio_val}
    if audio_fmt:
        aa["format"] = audio_fmt
    parts = [
        {"type": "input_audio", "input_audio": aa},
        {"type": "text", "text": prompt},
    ]
    result = await _cached_analyze(provider, parts, args)
    return json.dumps(result.to_dict(), ensure_ascii=False)


def _asr_default_prompt(lang: str) -> str:
    if lang == "zh":
        return "请将这段语音逐字转写为中文文本，只输出转写结果。"
    if lang == "en":
        return "Transcribe this audio verbatim to English text; output only the transcription."
    return "请转录这段语音内容，只输出转写结果。"


async def _analyze_asr(args: dict) -> str:
    provider = _get_provider()
    lang = args.get("language") or config.ASR_LANGUAGE
    audio_val, audio_fmt = await inputs.resolve_audio(
        args["audio"], max_size=config.MAX_ASR_SIZE, formats=config.ASR_FORMATS
    )
    aa = {"data": audio_val}
    if audio_fmt:
        aa["format"] = audio_fmt
    parts = [{"type": "input_audio", "input_audio": aa}]
    if not config.ASR_MODEL.endswith("-asr"):
        # 官方 -asr:content 仅 input_audio + asr_options;v2.5 顶替时需内置文本指令
        parts.append({"type": "text", "text": _asr_default_prompt(lang)})
    extra = {"asr_options": {"language": lang}}
    # 转写必须确定性:强制 temperature=0(忽略任何传入/默认值)
    args2 = dict(args)
    args2["temperature"] = 0.0
    result = await _cached_analyze(provider, parts, args2, model=config.ASR_MODEL, extra=extra)
    out = result.to_dict()
    out["language"] = lang
    out["model"] = config.ASR_MODEL
    return json.dumps(out, ensure_ascii=False)


async def _image_metadata(args: dict) -> str:
    src = args["image"]
    path = inputs.existing_path(src)
    if not path:
        raise InputError("image_metadata 需要本地文件路径或 file:// URL（EXIF 依赖原始文件）")
    info = await asyncio.to_thread(imageinfo.image_metadata, path)
    return json.dumps(info, ensure_ascii=False)


async def _cached_analyze(provider, parts: list[dict], args: dict, *,
                          model: str | None = None, extra: dict | None = None) -> VisionResult:
    max_tokens = _optional(args, "max_tokens")
    temperature = _optional(args, "temperature")
    mt = int(max_tokens) if max_tokens is not None else None
    tp = float(temperature) if temperature is not None else None

    if not config.CACHE_ENABLED:
        return await provider.analyze(parts, max_tokens=mt, temperature=tp, model=model, extra=extra)

    key_src = json.dumps(
        {"model": model or config.MODEL, "parts": parts, "max_tokens": max_tokens,
         "temperature": temperature, "extra": extra},
        ensure_ascii=False, sort_keys=True,
    )
    key = hashlib.sha256(key_src.encode("utf-8")).hexdigest()
    cache_file = os.path.join(config.CACHE_DIR, f"{key}.json")
    if os.path.isfile(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return VisionResult(**json.load(f))
        except Exception:
            pass

    result = await provider.analyze(parts, max_tokens=mt, temperature=tp, model=model, extra=extra)
    try:
        os.makedirs(config.CACHE_DIR, exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False)
    except Exception:
        pass
    return result


def _server_status() -> str:
    available = _provider is not None and _provider.is_available()
    return json.dumps(
        {
            "provider": "openai",
            "provider_available": available,
            "model": config.MODEL or None,
            "base_url_configured": bool(config.BASE_URL),
            "api_key_configured": bool(config.API_KEY),
            "video_mode": "local ffmpeg re-encode + cloud video_url",
            "max_tokens": config.MAX_TOKENS,
            "reasoning_effort": config.REASONING_EFFORT,
            "image": {
                "max_mb": _IMG_MAX_MB,
                "formats": sorted(config.IMAGE_FORMATS),
                "compress_default": config.IMAGE_COMPRESS,
                "max_pixels": config.IMAGE_MAX_PIXELS,
                "target_long_edge": config.IMAGE_TARGET_LONG_EDGE,
                "max_images_per_batch": config.MAX_IMAGES_PER_BATCH,
            },
            "video": {
                "max_mb": _VID_MAX_MB,
                "formats": sorted(config.VIDEO_FORMATS),
                "target_short_edge": config.VIDEO_TARGET_SHORT_EDGE,
                "target_fps": config.VIDEO_TARGET_FPS,
                "ffmpeg": config.VIDEO_FFMPEG,
            },
            "audio": {"max_mb": _AUD_MAX_MB, "formats": sorted(config.AUDIO_FORMATS)},
            "asr": {"max_mb": _ASR_MAX_MB, "model": config.ASR_MODEL, "language": config.ASR_LANGUAGE},
        },
        ensure_ascii=False,
    )


async def main():
    """Run the MCP Multimedia Server. Supports stdio and SSE transports."""
    _init_provider()
    if _provider is not None and _provider.is_available():
        print(f"[mcp-multimedia-server] provider ready (model={config.MODEL}).", file=sys.stderr)
    else:
        print(
            "[mcp-multimedia-server] provider not configured. "
            "Set MCP_MEDIA_BASE_URL / MCP_MEDIA_API_KEY / MCP_MEDIA_MODEL.",
            file=sys.stderr,
        )
    transport = os.getenv("MCP_TRANSPORT", "stdio")
    if transport == "sse":
        await _run_sse()
    else:
        await _run_stdio()


async def _run_stdio():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


async def _run_sse():
    try:
        from starlette.applications import Starlette
        from starlette.responses import Response
        from starlette.routing import Mount, Route as StarletteRoute
        import uvicorn
    except ImportError:
        print(
            "[mcp-multimedia-server] SSE transport requires starlette and uvicorn. "
            "Install with: pip install mcp-multimedia-server[sse]",
            file=sys.stderr,
        )
        sys.exit(1)

    host = os.getenv("MCP_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_PORT", "8093"))
    transport_instance = SseServerTransport("/messages/")

    async def handle_sse(request):
        async with transport_instance.connect_sse(
            request.scope, request.receive, request._send
        ) as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())
        return Response()

    app = Starlette(
        routes=[
            StarletteRoute("/sse", endpoint=handle_sse, methods=["GET"]),
            Mount("/messages/", app=transport_instance.handle_post_message),
        ]
    )
    print(f"[mcp-multimedia-server] SSE server starting on http://{host}:{port}", file=sys.stderr)
    config_srv = uvicorn.Config(app, host=host, port=port, log_level="info")
    await uvicorn.Server(config_srv).serve()


def run():
    """Synchronous entrypoint for the console script."""
    asyncio.run(main())


if __name__ == "__main__":
    run()
