#!/usr/bin/env python3
"""MCP Multimedia Server - Cloud image & video/audio analysis tools for LLM agents.

All media understanding is delegated to a cloud OpenAI-compatible multimodal model.
Video is handed to the model natively (``video_url``) and pure audio via ``input_audio``
(so the tool can do STT / speech summary) — there is no local frame extraction.
Configure the model via MCP_MEDIA_BASE_URL, MCP_MEDIA_API_KEY and MCP_MEDIA_MODEL.

Usage:
    mcp-multimedia-server                    # stdio (default)
    MCP_TRANSPORT=sse mcp-multimedia-server  # HTTP/SSE server
    MCP_PORT=8093 mcp-multimedia-server      # custom SSE port
"""

import asyncio
import hashlib
import json
import os
import sys
from typing import Any

from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.server.sse import SseServerTransport
from mcp.types import Tool, TextContent

from .providers.base import ProviderError, VisionResult
from .providers.openai_compat import OpenAICompatProvider
from .media import inputs, imageinfo
from .media.inputs import InputError
from .media.imageinfo import ImageInfoError

_ENV_FILE = os.path.join(os.path.expanduser("~"), ".config", "mcp-multimedia", ".env")
load_dotenv(_ENV_FILE)

PROVIDER_NAME = os.getenv("MCP_MEDIA_PROVIDER", "openai").strip()
BASE_URL = os.getenv("MCP_MEDIA_BASE_URL", "").strip()
API_KEY = os.getenv("MCP_MEDIA_API_KEY", "").strip()
MODEL = os.getenv("MCP_MEDIA_MODEL", "").strip()
CACHE_ENABLED = os.getenv("MCP_MEDIA_CACHE_ENABLED", "false").strip().lower() in ("1", "true", "yes")
CACHE_DIR = os.getenv("MCP_MEDIA_CACHE_DIR", "/tmp/mcp-vision-cache")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (ValueError, TypeError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (ValueError, TypeError):
        return default


MAX_TOKENS = _env_int("MCP_MEDIA_MAX_TOKENS", 1024)
TEMPERATURE = _env_float("MCP_MEDIA_TEMPERATURE", 0.2)
TIMEOUT = _env_float("MCP_MEDIA_TIMEOUT", 120.0)
MAX_RETRIES = _env_int("MCP_MEDIA_MAX_RETRIES", 3)
REASONING_EFFORT = os.getenv("MCP_MEDIA_REASONING_EFFORT", "").strip()

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
    factory = {
        "openai": lambda: OpenAICompatProvider(
            base_url=BASE_URL,
            api_key=API_KEY,
            model=MODEL,
            timeout=TIMEOUT,
            max_retries=MAX_RETRIES,
            default_max_tokens=MAX_TOKENS,
            default_temperature=TEMPERATURE,
            reasoning_effort=REASONING_EFFORT,
        ),
    }
    builder = factory.get(PROVIDER_NAME)
    _provider = builder() if builder else None


def _get_provider():
    if _provider is None:
        raise ProviderError(PROVIDER_NAME, f"Unknown or unconfigured provider '{PROVIDER_NAME}'.")
    if not _provider.is_available():
        raise ProviderError(
            PROVIDER_NAME,
            "Provider not configured. Set MCP_MEDIA_BASE_URL, MCP_MEDIA_API_KEY and MCP_MEDIA_MODEL.",
        )
    return _provider


_IMG_MAX_MB = int(inputs._max_size() / (1024 * 1024))
_VID_MAX_MB = int(inputs._max_video_size() / (1024 * 1024))
_AUD_MAX_MB = int(inputs._max_audio_size() / (1024 * 1024))

_IMAGE_PROP = {
    "type": "string",
    "description": (
        "图片输入，支持：本地绝对路径 / file:// URL / http(s) URL / base64(data URI)。"
        f"本地/内联图片最大 {_IMG_MAX_MB}MB；http(s) URL 由云端直接拉取，不受此限"
    ),
}
_VIDEO_PROP = {
    "type": "string",
    "description": (
        "视频输入（mp4/mov/webm 等），支持：http(s) URL(推荐) / 本地绝对路径 / file:// URL / base64(data URI)。"
        f"本地文件自动转 data URI，本地/内联视频最大 {_VID_MAX_MB}MB；http(s) URL 由云端直接拉取，不受此限"
    ),
}
_AUDIO_PROP = {
    "type": "string",
    "description": (
        "音频输入（mp3/wav/ogg/opus/flac/m4a/aac），支持：http(s) URL / 本地绝对路径 / file:// URL / base64(data URI)。"
        f"注意：input_audio 不支持 URL 直传——音频 URL 由本机 MCP 服务下载后转 base64 内嵌进请求，故 URL 与本地文件同受最大 {_AUD_MAX_MB}MB 限制"
    ),
}


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="analyze_image",
            description=f"图片理解问答：调用云端视觉大模型对图片进行描述、问答、OCR、图表/UI/报错分析等。可用 prompt 自由提问，或用 preset 选择任务类型。本地/内联图片最大 {_IMG_MAX_MB}MB，http(s) URL 不受此限。",
            inputSchema={
                "type": "object",
                "properties": {
                    "image": _IMAGE_PROP,
                    "prompt": {"type": "string", "description": "自由指令/问题；提供后覆盖 preset", "default": ""},
                    "preset": {
                        "type": "string",
                        "enum": list(PROMPT_PRESETS.keys()),
                        "description": "任务预设：describe 描述 / ocr 文字识别 / chart 图表 / ui 界面 / diagram 示意图 / error 报错诊断",
                        "default": "describe",
                    },
                    "max_tokens": {"type": "integer", "description": "可选，本次生成上限（默认 65536）"},
                    "temperature": {"type": "number", "description": "可选，采样温度"},
                },
                "required": ["image"],
            },
        ),
        Tool(
            name="analyze_video",
            description=f"视频理解：将视频原生交给云端多模态大模型处理——抽帧与时序对齐由模型服务端完成，本地不做任何处理。视频走 video_url（URL 或 base64 data URI 均有效；本地文件自动转 data URI）。本地/内联视频最大 {_VID_MAX_MB}MB，http(s) URL 不受此限。当前模型与配置可用 get_server_status 查询。",
            inputSchema={
                "type": "object",
                "properties": {
                    "video": _VIDEO_PROP,
                    "prompt": {"type": "string", "description": "自由指令/问题，如'这段视频里发生了什么？按时间顺序描述'", "default": ""},
                    "max_tokens": {"type": "integer", "description": "可选，本次生成上限（默认 65536）"},
                    "temperature": {"type": "number", "description": "可选，采样温度"},
                },
                "required": ["video"],
            },
        ),
        Tool(
            name="analyze_audio",
            description=f"语音理解/STT：将纯音频原生交给云端多模态大模型进行转录与语音总结——语音识别由模型服务端完成，本地不做任何处理。音频走 input_audio（只接受内联 base64，不接受 URL）；http(s) URL 由本机 MCP 服务下载后转 base64 内嵌，URL 与本地文件同受最大 {_AUD_MAX_MB}MB 限制。当前模型与配置可用 get_server_status 查询。",
            inputSchema={
                "type": "object",
                "properties": {
                    "audio": _AUDIO_PROP,
                    "prompt": {"type": "string", "description": "自由指令/问题，如'转录这段语音'；不传则默认转录并总结主旨", "default": ""},
                    "max_tokens": {"type": "integer", "description": "可选，本次生成上限（默认 65536）"},
                    "temperature": {"type": "number", "description": "可选，采样温度"},
                },
                "required": ["audio"],
            },
        ),
        Tool(
            name="image_metadata",
            description="图片元信息：读取尺寸、格式、颜色模式与 EXIF（含拍摄时间、GPS 若有），本地 Pillow 解析，不调用云端。",
            inputSchema={
                "type": "object",
                "properties": {"image": _IMAGE_PROP},
                "required": ["image"],
            },
        ),
        Tool(
            name="get_server_status",
            description="查询服务状态：云端视觉模型是否已配置、当前模型与配置项。",
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
    image_url = await inputs.resolve_image_url(args["image"])
    parts = [
        {"type": "image_url", "image_url": {"url": image_url}},
        {"type": "text", "text": prompt},
    ]
    result = await _cached_analyze(provider, parts, args)
    out = result.to_dict()
    out["preset"] = preset if not args.get("prompt") else None
    return json.dumps(out, ensure_ascii=False)


async def _analyze_video(args: dict) -> str:
    provider = _get_provider()
    prompt = args.get("prompt") or (
        "请按时间顺序详细描述这个视频的内容：场景、主要对象、发生的动作与事件。"
    )
    video_url = await inputs.resolve_video_url(args["video"])
    parts = [
        {"type": "video_url", "video_url": {"url": video_url}},
        {"type": "text", "text": prompt},
    ]
    result = await _cached_analyze(provider, parts, args)
    return json.dumps(result.to_dict(), ensure_ascii=False)


async def _analyze_audio(args: dict) -> str:
    provider = _get_provider()
    prompt = args.get("prompt") or "请转录这段语音的内容，并简要总结其主旨。"
    audio_data, audio_fmt = await inputs.resolve_audio(args["audio"])
    parts = [
        {"type": "input_audio", "input_audio": {"data": audio_data, "format": audio_fmt}},
        {"type": "text", "text": prompt},
    ]
    result = await _cached_analyze(provider, parts, args)
    return json.dumps(result.to_dict(), ensure_ascii=False)


async def _image_metadata(args: dict) -> str:
    src = args["image"]
    path = inputs.existing_path(src)
    if not path:
        raise InputError(
            "image_metadata 需要本地文件路径或 file:// URL（EXIF 依赖原始文件）"
        )
    info = await asyncio.to_thread(imageinfo.image_metadata, path)
    return json.dumps(info, ensure_ascii=False)


async def _cached_analyze(provider, parts: list[dict], args: dict) -> VisionResult:
    max_tokens = _optional(args, "max_tokens")
    temperature = _optional(args, "temperature")
    mt = int(max_tokens) if max_tokens is not None else None
    tp = float(temperature) if temperature is not None else None

    if not CACHE_ENABLED:
        return await provider.analyze(parts, max_tokens=mt, temperature=tp)

    key_src = json.dumps(
        {"model": MODEL, "parts": parts, "max_tokens": max_tokens, "temperature": temperature},
        ensure_ascii=False, sort_keys=True,
    )
    key = hashlib.sha256(key_src.encode("utf-8")).hexdigest()
    cache_file = os.path.join(CACHE_DIR, f"{key}.json")
    if os.path.isfile(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return VisionResult(**json.load(f))
        except Exception:
            pass

    result = await provider.analyze(parts, max_tokens=mt, temperature=tp)
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False)
    except Exception:
        pass
    return result


def _server_status() -> str:
    available = _provider is not None and _provider.is_available()
    return json.dumps(
        {
            "provider": PROVIDER_NAME,
            "provider_available": available,
            "model": MODEL or None,
            "base_url_configured": bool(BASE_URL),
            "api_key_configured": bool(API_KEY),
            "video_mode": "cloud-native (no local frame extraction)",
            "cache_enabled": CACHE_ENABLED,
            "max_tokens": MAX_TOKENS,
            "allowed_image_formats": sorted(inputs._allowed_formats()),
            "max_image_size": inputs._max_size(),
            "max_video_size": inputs._max_video_size(),
        },
        ensure_ascii=False,
    )


async def main():
    """Run the MCP Multivision Server. Supports stdio and SSE transports."""
    _init_provider()

    if _provider is not None and _provider.is_available():
        print(f"[mcp-multimedia-server] Vision provider '{PROVIDER_NAME}' ready (model={MODEL}).", file=sys.stderr)
    else:
        print(
            "[mcp-multimedia-server] Vision provider not configured. "
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
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server_instance = uvicorn.Server(config)
    await server_instance.serve()


def run():
    """Synchronous entrypoint for the console script."""
    asyncio.run(main())


if __name__ == "__main__":
    run()
