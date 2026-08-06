"""Hardcoded configuration (no .env / config file).

上游三件套(BASE_URL / API_KEY / MODEL)从进程环境读取(.claude.json 注册时传入,
CC Switch 切换用);其余所有限制/处理参数硬编码在此,改这里即可调整。
"""

import os

# --- 上游(env 覆盖;API key 只来自环境,不入库) ---
BASE_URL = os.getenv("MCP_MEDIA_BASE_URL", "https://opencode.ai/zen/go/v1").strip()
API_KEY = os.getenv("MCP_MEDIA_API_KEY", "").strip()
MODEL = os.getenv("MCP_MEDIA_MODEL", "mimo-v2.5").strip()

# --- 输出 / 推理 ---
MAX_TOKENS = 128000            # 单次输出上限,对齐 opencode 注册表 maxTokens=128000(opencode-go/mimo-v2.5);
                               # 大文档OCR输出可能很大,极端长输出可按次传更高;总约束 输入+输出≤1M上下文
TEMPERATURE = 0.2
TIMEOUT = 120.0
MAX_RETRIES = 3
REASONING_EFFORT = "none"          # 关推理,省输出长度
CACHE_ENABLED = False
CACHE_DIR = "/tmp/mcp-media-cache"

# --- 大小上限(字节;本地/内联输入) ---
MAX_IMAGE_SIZE = 50 * 1024 * 1024   # 官方:图片 base64 ≤50MB
MAX_VIDEO_SIZE = 50 * 1024 * 1024   # 官方:视频 base64 ≤50MB
MAX_AUDIO_SIZE = 50 * 1024 * 1024   # 官方:音频理解 base64 ≤50MB(URL ≤100MB 走服务端)
MAX_ASR_SIZE = 10 * 1024 * 1024     # 官方:ASR base64 ≤10MB

# --- 图片处理 ---
IMAGE_COMPRESS = True               # 默认压缩(OCR/文档解析,省 token)
IMAGE_MAX_PIXELS = 8_388_608        # 8MP 硬上限(官方 IMAGE_MAX_PIXELS)
IMAGE_TARGET_LONG_EDGE = 1600       # 压缩目标长边 px(≈A4@150DPI,~2.5MP)
IMAGE_JPEG_QUALITY = 85
MAX_IMAGES_PER_BATCH = 100          # 批量图片最大张数(固定值)。依据:压缩A4图~1.8K token/张→1M窗口约400张;8MP图~8K token/张→约125张;取100安全余量,实测100张网关稳定

# --- 视频重编码(本地 ffmpeg) ---
VIDEO_TARGET_SHORT_EDGE = 720       # 重编码目标短边(720p)
VIDEO_TARGET_FPS = 2.0              # 默认帧率(范围 [1,30])
VIDEO_FFMPEG = "ffmpeg"

# --- 格式白名单(官方文档) ---
IMAGE_FORMATS = {"jpeg", "png", "gif", "webp", "bmp"}
VIDEO_FORMATS = {"mp4", "mov", "avi", "wmv"}
AUDIO_FORMATS = {"mp3", "wav", "flac", "m4a", "ogg"}

# --- ASR ---
ASR_MODEL = "mimo-v2.5"             # 网关暂无 -asr,先用 v2.5 顶替;切官方 API 后改 mimo-v2.5-asr
ASR_LANGUAGE = "auto"               # auto / zh / en
ASR_FORMATS = {"wav", "mp3"}        # 官方 ASR 仅支持 wav/mp3(区别于音频理解的 5 种)
