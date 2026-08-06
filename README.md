# MCP Multimedia Server

**给 LLM agent(Claude Code / Codex 等)外挂小米 MiMo V2.5 的多模态理解能力**——图片、视频、纯音频、语音转写(ASR),全部走云端 MiMo V2.5 完成,本地只做必要的预处理(图片压缩、视频重编码)以省 token / 控制体积。

MCP(MCP = Model Context Protocol)stdio 服务器,注册进 agent 后即可当工具直接调用。

## 用途

- **图片理解 / OCR**:单张或批量(一次最多 100 张,适合文档逐页解析)
- **视频理解**:本地视频自动压缩重编码后发送,`fps` 可调精细度
- **音频理解 / STT**:转录或总结一段语音
- **语音转写(ASR)**:纯转写,支持语种指定
- 底层统一使用 **MiMo V2.5**(原生多模态,1M 上下文),通过任意 OpenAI 兼容端点接入

## 工具

| 工具 | 输入 | 说明 |
|------|------|------|
| `analyze_image` | `image`(单张)或 `image_batch`(批量数组) | 描述/OCR/图表/UI/报错诊断;本地图片默认压缩(8MP 上限 + A4-OCR 分辨率,`compress=false` 可关);批量最多 100 张 |
| `analyze_video` | `video` + `fps`(1-30) | 本地视频自动重编码(短边 720p);支持 URL |
| `analyze_audio` | `audio` | 音频理解/STT(转录+总结);推荐本地文件或 base64 |
| `asr` | `audio` + `language`(auto/zh/en) | 纯语音转写;仅 wav/mp3,≤10MB |
| `image_metadata` | 本地路径 | 读取图片尺寸/格式/EXIF/GPS,不调用云端 |
| `get_server_status` | — | 查看当前模型与各项限制/参数 |

## 工作原理

```
agent ── MCP stdio ──> 本服务 ── OpenAI 兼容端点 ──> MiMo V2.5(云端)
                        │
                        ├─ 图片:本地压缩(8MP 上限 + A4-OCR 分辨率)→ image_url
                        ├─ 视频:ffmpeg 重编码(短边720p + fps)→ video_url
                        └─ 音频:本地/base64 → input_audio(data + format)
```

- 本地处理仅在**发送前预处理**,不做任何媒体理解
- 所有推理/识别/转写在云端 MiMo V2.5 完成
- 上下文窗口 1M tokens;媒体按分辨率/时长计 token(压缩后 A4 图约 1.8K token/张)

## 环境要求

- Python 3.10+
- `ffmpeg`(视频重编码用,须在 PATH)
- 一个 OpenAI 兼容的多模态端点 + API key(如小米 MiMo 官方 `api.xiaomimimo.com`,或任意代理网关)

## 安装

```bash
# 1. 建 venv 并从 GitHub 安装
uv venv ~/.local/venvs/mcp-multimedia-server
uv pip install --python ~/.local/venvs/mcp-multimedia-server \
    git+https://github.com/ThermalEng/mcp-multimedia-server.git

# 2. ffmpeg(视频重编码)
sudo apt install ffmpeg
```

## 配置

操作参数(大小上限、压缩目标、视频重编码、格式白名单等)硬编码在 `mcp_multimedia_server/config.py`,改代码即可调整。上游三件套从进程环境读取:

| 环境变量 | 说明 |
|------|------|
| `MCP_MEDIA_BASE_URL` | OpenAI 兼容端点(默认 `https://opencode.ai/zen/go/v1`) |
| `MCP_MEDIA_API_KEY` | API key(不写进代码/仓库,只放环境) |
| `MCP_MEDIA_MODEL` | 模型名(默认 `mimo-v2.5`) |

## Claude Code 注册(stdio)

把下面合并到 `~/.claude.json` 的 `mcpServers`:

```json
{
  "mcpServers": {
    "multimedia": {
      "type": "stdio",
      "command": "~/.local/venvs/mcp-multimedia-server/bin/mcp-multimedia-server",
      "args": [],
      "env": {
        "MCP_MEDIA_BASE_URL": "https://opencode.ai/zen/go/v1",
        "MCP_MEDIA_API_KEY": "sk-你的key",
        "MCP_MEDIA_MODEL": "mimo-v2.5"
      }
    }
  }
}
```

注册后重启 agent,即可用 `analyze_image` / `analyze_video` / `analyze_audio` / `asr` 等工具。

## 备注

- 依赖 `mcp` SDK 须钉在 `<2`(SDK 2.x 移除了 `Server.list_tools` 装饰器)
- 视频重编码依赖系统 `ffmpeg`(`config.VIDEO_FFMPEG` 可指绝对路径)
- 专用模型说明:`mimo-v2.5-asr` / `mimo-v2.5-tts` 为官方独立模型;若你的端点未开放它们(部分网关 401),ASR 会用 `mimo-v2.5` 顶替(仍可转写),TTS 则不可用
- 流式/SSE 传输:`MCP_TRANSPORT=sse` / `MCP_PORT=8093`(需 `pip install mcp-multimedia-server[sse]`)
