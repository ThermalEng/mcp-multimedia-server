# MCP Multimedia Server

通过 OpenAI 兼容多模态端点,为 LLM agent 提供**图片 / 视频 / 纯音频**云端分析能力的 MCP 服务器。视频抽帧、时序对齐、语音识别全部由云端模型完成,本地不做任何媒体处理。

## 工具

| 工具 | 输入 | 说明 |
|------|------|------|
| `analyze_image` | 本地路径 / URL / base64 | 描述、OCR、图表、UI、报错诊断(可 `preset` 或自由 `prompt`) |
| `analyze_video` | 本地路径 / URL / base64 | 视频理解,走 `video_url`(URL 或 data URI 均有效) |
| `analyze_audio` | 本地路径 / URL / base64 | STT 转录 / 语音总结,走 `input_audio`(内联 base64) |
| `image_metadata` | 本地路径 | 本地 Pillow 读取尺寸/格式/EXIF,不调用云端 |
| `get_server_status` | — | 当前模型、max_tokens、文件上限等配置 |

所有工具可选参数:`prompt`、`max_tokens`(默认 65536)、`temperature`。

## 配置

启动时按以下优先级取值(高→低):
1. 进程环境变量(即 Claude Code MCP 配置里的 `env`)
2. `~/.config/mcp-multimedia/.env`(服务端 `load_dotenv` 自动加载,含 API key,权限建议 600)
3. 代码内置默认值

主要环境变量:

| 变量 | 默认 | 说明 |
|------|------|------|
| `MCP_MEDIA_BASE_URL` | — | OpenAI 兼容端点 |
| `MCP_MEDIA_API_KEY` | — | API key |
| `MCP_MEDIA_MODEL` | — | 模型名(如 `mimo-v2.5`) |
| `MCP_MEDIA_MAX_TOKENS` | 1024 | 单次生成上限(建议调大,1024 会被推理耗完导致空输出) |
| `MCP_MEDIA_REASONING_EFFORT` | *(空)* | 设为 `none` 可关闭推理,`reasoning_tokens=0` |
| `MCP_MEDIA_MAX_IMAGE_SIZE` | 20MB | 本地/内联图片上限(字节) |
| `MCP_MEDIA_MAX_VIDEO_SIZE` | 100MB | 本地/内联视频上限(字节) |
| `MCP_MEDIA_MAX_AUDIO_SIZE` | 20MB | 音频上限(独立;URL 也会被下载后内联,故同样受限) |

## 媒体内容类型实测(OpenCode Go 端点 / MiMo-V2.5)

| 内容块类型 | 视频 | 音频 |
|------|------|------|
| `video_url`(URL 或 base64 data URI) | ✅ 有效 | ❌ 上游 400 |
| `audio_url` | — | ⚠️ 200 但音频不送达 |
| `input_audio`(内联 base64 + format) | — | ✅ 有效(STT) |
| `input_video` / `video`(data) | ❌ 200 但不送达(会幻觉) | — |

因此:视频走 `video_url`(URL 直传、云端拉取,不受本地上限);音频走 `input_audio`(**不支持 URL 直传**,URL 由本机服务下载后转 base64 内嵌,故受 `MCP_MEDIA_MAX_AUDIO_SIZE` 限制)。

## 安装

```bash
# 从仓库安装(推荐装进独立 venv)
uv venv ~/.local/venvs/mcp-multimedia-server
uv pip install --python ~/.local/venvs/mcp-multimedia-server git+https://github.com/ThermalEng/mcp-multimedia-server.git
```

## Claude Code 注册(stdio)

```json
{
  "mcpServers": {
    "multimedia": {
      "type": "stdio",
      "command": "/home/mc/.local/venvs/mcp-multimedia-server/bin/mcp-multimedia-server",
      "args": [],
      "env": {
        "MCP_MEDIA_BASE_URL": "https://opencode.ai/zen/go/v1",
        "MCP_MEDIA_API_KEY": "sk-...",
        "MCP_MEDIA_MODEL": "mimo-v2.5"
      }
    }
  }
}
```

## 备注

- 依赖 `mcp` SDK 须钉在 `<2`(SDK 2.x 移除了 `Server.list_tools` 装饰器,与该项目不兼容)。
- SSE 传输:`MCP_TRANSPORT=sse` / `MCP_PORT=8093`(需 `pip install mcp-multimedia-server[sse]`)。
