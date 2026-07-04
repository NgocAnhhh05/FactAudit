# TurboQuant+ Integration Guide for FACT-AUDIT (v2)

**Version:** 2.0
**Date:** 2026-06-28
**Author:** AI Systems Engineer
**Thay đổi chính so với v1:** **Loại bỏ hoàn toàn Ollama.** Model được tải trực tiếp về máy dạng file GGUF trong `Factaudit/models/` và được serve bởi `llama-cpp-turboquant`. Client (Factaudit) chỉ giao tiếp qua OpenAI-compatible API (`ChatOpenAI`).

---

## Table of Contents

1. [Tóm tắt thay đổi v1 → v2 (What Changed)](#1-tóm-tắt-thay-đổi-v1--v2-what-changed)
2. [Architecture Overview](#2-tổng-quan-kiến-trúc-architecture-overview)
3. [Server Setup Guide](#3-hướng-dẫn-thiết-lập-server-llama-cpp-turboquant-setup)
4. [Code Implementation (trong Factaudit)](#4-code-implementation-trong-factaudit)
5. [Script khởi động (Bash)](#5-script-khởi-động-bash)
6. [Usage Examples](#6-cách-sử-dụng-usage-examples)
7. [Performance Comparison](#7-so-sánh-hiệu-năng-performance-comparison)
8. [Troubleshooting](#8-xử-lý-sự-cố-troubleshooting)
9. [Appendix](#9-appendix)

---

## 1. Tóm tắt thay đổi v1 → v2 (What Changed)

| Khía cạnh                    | v1 (cũ)                                                             | v2 (mới)                                                                                         |
| ------------------------------ | -------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| **Nguồn model**         | Qua Ollama (`ollama pull llama3.1`), chạy daemon `ollama serve` | File GGUF tải trực tiếp về`Factaudit/models/` (vd: `Qwen3-14B-Q8_0.gguf`)                 |
| **Client LangChain**     | `ChatOllama` (fallback) + `ChatOpenAI`                           | **Chỉ `ChatOpenAI`** (gọi `llama-cpp-turboquant`)                                     |
| **Thư viện Python**    | `langchain-ollama` (có)                                           | **Đã xoá `langchain-ollama`**, thêm `langchain-openai`                              |
| **Fallback**             | `ENABLE_OLLAMA_FALLBACK` → Ollama                                 | Không còn Ollama. Cloud fallback`ChatGoogleGenerativeAI` (tuỳ chọn, qua `GEMINI_API_KEY`) |
| **Biến env Ollama**     | `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `ENABLE_OLLAMA_FALLBACK`    | **Đã xoá**. Thay bằng `GGUF_MODEL_PATH` / `GGUF_MODEL_FILE`                         |
| **Server**               | `llama-cpp-turboquant` qua OpenAI API                              | **Giữ nguyên** (chính là trọng tâm)                                                   |
| **Khởi động server**  | Gõ lệnh`./bin/server ...` thủ công                             | Script`scripts/start_server.sh [baseline\|turboquant\|both]`                                      |
| **`_print_mode_info`** | In`Model: Llama-3-8B-Instruct` (tên model Ollama)                 | In đường dẫn file GGUF + trạng thái tồn tại                                               |

> **Điểm quan trọng:** `llama-cpp-turboquant` (server C++) và `turboquant_plus` (thuật toán nén) **không đổi**. Toàn bộ logic nén KV cache (PolarQuant / QJL / turbo3-turbo4) vẫn nằm trong server. Factaudit **chỉ thay đổi lớp client**: từ Ollama sang ChatOpenAI + quản lý file GGUF cục bộ.

---

## 2. Tổng quan Kiến trúc (Architecture Overview)

### 2.1 Nguyên lý hoạt động

Hệ thống vẫn theo kiến trúc **Client-Server** tách biệt. Điểm khác biệt cốt lõi: **model nằm trong thư mục `Factaudit/models/`** và được `llama-cpp-turboquant` load trực tiếp. Không còn daemon Ollama đứng giữa.

```
┌─────────────────────────────────────────────────────────────────────┐
│                       FACT-AUDIT (Client Layer)                      │
│                                                                      │
│   Factaudit/                                                         │
│   ├── src/config.py     LLMFactory → ChatOpenAI (mode switching)     │
│   ├── src/main.py        Entry point / orchestrator                  │
│   └── models/            ◂── KHO GGUF CỤC BỘ ──▸                     │
│       └── Qwen3-14B-Q8_0.gguf   (tải về 1 lần, dùng cho cả 2 mode)    │
│                                                                      │
│   ┌────────────────────────────────────────────┐                     │
│   │   LangGraph Orchestrator (5 agents)         │                     │
│   │   Appraiser · Inquirer · Quality · Eval ·   │                     │
│   │   Prober + Target (model under test)        │                     │
│   └────────────────────────────────────────────┘                     │
│                      │                                               │
│   ┌──────────────────▼─────────────────────────┐                     │
│   │   LLMFactory (src/config.py)                │                     │
│   │   - resolve_gguf_model_path()  ← log path   │                     │
│   │   - mode = baseline (8080) | turboquant     │                     │
│   │   - ChatOpenAI(base_url=<api_base>)         │                     │
│   └──────────────────┬──────────────────────────┘                     │
│                      │ base_url trỏ tới 8080 hoặc 8081               │
└──────────────────────┼──────────────────────────────────────────────┘
                       │  OpenAI-compatible API (REST /v1/chat/completions)
                       │  KHÔNG CÒN OLLAMA ở bất kỳ đâu
        ┌──────────────▼───────────────────────────┐
        │   llama-cpp-turboquant  (2 server C++)   │
        │                                          │
        │   ┌────────────────────────────────────┐ │
        │   │ Baseline Server   :8080            │ │     load cùng 1 file GGUF
        │   │  - cache-type-k/v = f32            │ │  ┌──────────────────┐
        │   │  - ctx-size 8192                   │ │  │ Factaudit/models │
        │   └────────────────────────────────────┘ │  │ Qwen3-14B-       │
        │   ┌────────────────────────────────────┐ │◀─│  Q8_0.gguf       │
        │   │ TurboQuant+ Server :8081           │ │  └──────────────────┘
        │   │  - cache-type-k = turbo3           │ │
        │   │  - cache-type-v = turbo4           │ │
        │   │  - ctx-size 32768 (4× baseline)    │ │
        │   └────────────────────────────────────┘ │
        └──────────────────────────────────────────┘
                       │
        ┌──────────────▼───────────────────────────┐
        │   turboquant_plus (Optional)             │
        │   - Training & Calibration Scripts       │
        │   - Compression Algorithms (chạy 1 lần)  │
        │   - KHÔNG cần ở runtime                  │
        └──────────────────────────────────────────┘
```

### 2.2 Thành phần hệ thống

| Thành phần                            | Vai trò                                                                                | Chạy tại Runtime | Cần build?                         |
| --------------------------------------- | --------------------------------------------------------------------------------------- | ------------------ | ----------------------------------- |
| **Factaudit** (Client)            | Python LangGraph orchestrator, 5 agent fact-checking,`LLMFactory` tạo `ChatOpenAI` | ✅ Yes             | `pip install -r requirements.txt` |
| **Factaudit/models/**             | Lưu file GGUF tải về (vd`Qwen3-14B-Q8_0.gguf`)                                     | ✅ (file tĩnh)    | Tải về (xem §3.2)                |
| **llama-cpp-turboquant** (Server) | C++ server, load GGUF + serve inference với KV cache compression                       | ✅ Yes             | Build CMake (xem §3.1)             |
| **turboquant_plus**               | Thuật toán nén, training/calibration (1 lần)                                        | ❌ No              | Tuỳ chọn                          |

**Điểm quan trọng:** Code Factaudit **KHÔNG chứa** logic tính toán của TurboQuant+. Nó cũng **KHÔNG còn phụ thuộc Ollama**. Factaudit chỉ cần: (1) file GGUF trong `models/`, và (2) server `llama-cpp-turboquant` đang chạy. Giao tiếp hoàn toàn qua **OpenAI-compatible API**.

### 2.3 Lưu lượng dữ liệu (Data Flow)

```
User Query
     │
     ▼
Factaudit Orchestrator (main.py)
     │
     ├──► LLMFactory tạo ChatOpenAI theo mode
     │        │
     │        ├──► MODE=baseline    → BASELINE_API_BASE    (http://localhost:8080/v1)
     │        └──► MODE=turboquant  → TURBOQUANT_API_BASE  (http://localhost:8081/v1)
     │
     ▼
API Request  (POST /v1/chat/completions, OpenAI format)
     │   field "model" = MODEL_NAME (alias, vd "Qwen3-14B-Q8_0")
     ▼
llama-cpp-turboquant Server  (đã load Factaudit/models/*.gguf)
     │
     ├──► Baseline :8080  → KV cache f32        (chuẩn, không nén)
     └──► TurboQuant :8081 → KV cache turbo3/turbo4 (nén)
     │
     ▼
Response  →  Factaudit xử lý  →  Fact-Checking Result
```

---

## 3. Hướng dẫn thiết lập Server (llama-cpp-turboquant Setup)

### 3.1 Yêu cầu hệ thống

| Yêu cầu | Tối thiểu                                           | Khuyến nghị                          |
| --------- | ----------------------------------------------------- | -------------------------------------- |
| CPU       | 8 cores                                               | 16+ cores                              |
| RAM       | 16 GB                                                 | 32 GB+                                 |
| GPU VRAM  | 8 GB (Baseline)                                       | 12 GB+ (TurboQuant, vì ctx lớn hơn) |
| OS        | Linux / Windows (Git Bash / WSL)                      | Ubuntu 22.04 LTS                       |
| Compiler  | GCC 11+ (hoặc MSVC/CMake trên Windows)              | GCC 13+                                |
| Disk      | ≥ kích thước file GGUF (vd Qwen3-14B Q8 ≈ 15 GB) | SSD                                    |

### 3.2 Chuẩn bị file GGUF (MỚI so với v1)

Vì không dùng Ollama, bạn **tải trực tiếp** file GGUF về thư mục `Factaudit/models/`.

```bash
cd Factaudit
mkdir -p models
cd models

# Ví dụ: Qwen3-14B lượng tử Q8_0
# (thay URL bằng nguồn bạn tin tưởng, vd HuggingFace)
# curl -L -o Qwen3-14B-Q8_0.gguf "https://huggingface.co/.../resolve/main/Qwen3-14B-Q8_0.gguf"

# Kiểm tra
ls -lh models/
# -rw-r--r-- ... 15G ... Qwen3-14B-Q8_0.gguf
```

> Cấu trúc thư mục mong đợi:
>
> ```
> Factaudit/
> ├── src/
> │   └── config.py
> ├── models/                ← THÊM ở v2
> │   └── Qwen3-14B-Q8_0.gguf
> ├── scripts/
> │   └── start_server.sh    ← THÊM ở v2
> └── .env
> ```

File này dùng **chung cho cả 2 server** (baseline + turboquant). Hai server chỉ khác nhau ở cấu hình KV cache & context khi khởi động — không cần 2 file model.

### 3.3 Build llama-cpp-turboquant

```bash
# Clone repository (cạnh Factaudit)

cd llama-cpp-turboquant

# Build với hỗ trợ TurboQuant + CUDA (nếu có GPU NVIDIA)
mkdir build && cd build
cmake .. -DLLAMA_CUBLAS=ON -DLLAMA_TURBOQUANT=ON
cmake --build . --config Release -j$(nproc)

# Verify build
./bin/llama-server --help | grep -i turbo     # bản mới
# hoặc
./bin/server --help | grep -i turbo            # bản build cũ
```

> **Lưu ý tên binary:** bản llama.cpp mới đặt tên là `llama-server`, bản cũ là `server`. Script `start_server.sh` mặc định tìm `<repo>/build/bin/llama-server`. Nếu bản của bạn là `server`, hãy `export LLAMA_SERVER_BIN=/đường/dẫn/build/bin/server`.

### 3.4 Khởi động server bằng script (KHUYẾN NGHỊ)

Từ v2, dùng script tham số hóa thay vì gõ tay (xem chi tiết §5):

```bash
cd Factaudit

# Chỉ Baseline (foreground, port 8080, cache f32)
./scripts/start_server.sh baseline

# Chỉ TurboQuant+ (foreground, port 8081, cache turbo3/turbo4, ctx 32768)
./scripts/start_server.sh turboquant

# Cả 2 cùng lúc (background, ghi log vào logs/)
./scripts/start_server.sh both
```

Script tự kiểm tra tồn tại của binary server và file GGUF, trỏ `--model` về `Factaudit/models/`, và ghi log vào `logs/server_<mode>.log`.

### 3.5 Khởi động server bằng lệnh tay (tương đương v1)

Nếu muốn chạy trực tiếp (không qua script):

**Baseline (port 8080, cache f32):**

```bash
cd llama-cpp-turboquant/build

./bin/llama-server \
    --host 0.0.0.0 \
    --port 8080 \
    --model /đường/đến/Factaudit/models/Qwen3-14B-Q8_0.gguf \
    --alias Qwen3-14B-Q8_0 \
    --threads 8 \
    --n-gpu-layers 35 \
    --ctx-size 8192 \
    --batch-size 512 \
    --ubatch-size 128 \
    --cache-type-k f32 \
    --cache-type-v f32 \
    --metrics
```

**TurboQuant+ (port 8081, cache turbo3/turbo4, ctx 32768):**

```bash
cd llama-cpp-turboquant/build

./bin/llama-server \
    --host 0.0.0.0 \
    --port 8081 \
    --model /đường/đến/Factaudit/models/Qwen3-14B-Q8_0.gguf \
    --alias Qwen3-14B-Q8_0 \
    --threads 8 \
    --n-gpu-layers 35 \
    --ctx-size 32768 \
    --batch-size 1024 \
    --ubatch-size 256 \
    --cache-type-k turbo3 \
    --cache-type-v turbo4 \
    --metrics
```

> `--alias` nên đặt bằng `MODEL_NAME` trong `.env` (mặc định `Qwen3-14B-Q8_0`) để endpoint `/v1/models` khớp với trường `model` mà `ChatOpenAI` gửi đi.

### 3.6 So sánh cấu hình 2 Mode

| Tham số                             | Baseline  | TurboQuant+ | Giải thích          |
| ------------------------------------ | --------- | ----------- | --------------------- |
| `--port`                           | 8080      | 8081        | Cổng API             |
| `--cache-type-k`                   | f32       | turbo3      | Format cache Key      |
| `--cache-type-v`                   | f32       | turbo4      | Format cache Value    |
| `--ctx-size`                       | 8192      | 32768       | Context size (tokens) |
| `--turbo-quant-bits`               | N/A       | 4           | Bits cho quantization |
| `--turbo-group-size`               | N/A       | 64          | Group size cho QJL    |
| `--batch-size` / `--ubatch-size` | 512 / 128 | 1024 / 256  | Batch prompt          |
| VRAM ước tính                     | ~8 GB     | ~3.2 GB     | Tiết kiệm ~60%      |

### 3.7 Kiểm tra cả 2 server

```bash
# Danh sách model (phải trả về alias Qwen3-14B-Q8_0)
curl http://localhost:8080/v1/models
curl http://localhost:8081/v1/models

# Health check
curl -s http://localhost:8080/health && echo "  - Baseline OK"
curl -s http://localhost:8081/health && echo "  - TurboQuant OK"

# Test inference nhanh (chú ý "model" phải khớp --alias / MODEL_NAME)
curl -X POST http://localhost:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-14B-Q8_0",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "max_tokens": 50
  }'
```

---

## 4. Code Implementation (trong Factaudit)

> Khác v1 (đề xuất tách thành `src/config/`, `src/llm/` nhiều file), **code thực tế giữ một file duy nhất `src/config.py`** với class `LLMFactory`. Phần này mô tả đúng code đang triển khai.

### 4.1 `src/config.py` — bản v2 (đã gỡ Ollama)

Các thay đổi cốt lõi:

1. **Xoá** `from langchain_ollama import ChatOllama`, method `_create_fallback_ollama`, biến `ENABLE_OLLAMA_FALLBACK` và toàn bộ logic Ollama.
2. **Thêm** `resolve_gguf_model_path()` resolve đường dẫn GGUF trong `Factaudit/models/` (ưu tiên `GGUF_MODEL_PATH` → `GGUF_MODEL_FILE` → scan `*.gguf` → default).
3. `LLMFactory._create_base_llm()` trả về `ChatOpenAI(base_url=self._api_base, ...)`. `base_url` là `BASELINE_API_BASE` (8080) hoặc `TURBOQUANT_API_BASE` (8081) tuỳ mode.
4. `_print_mode_info()` in **đường dẫn file GGUF** + trạng thái tồn tại thay vì tên model Ollama.
5. **Giữ nguyên** tham số `format="json"` của `create_judge()` và toàn bộ signature của 4 hàm `create_*` → không phá vỡ `llm_*.with_structured_output(PydanticModel)` ở các agent.

Toàn bộ file `src/config.py` (v2):

```python
"""
===============================================
FACT-AUDIT Configuration Module  (v2 - No Ollama)
===============================================
Module này quản lý việc khởi tạo LLM instances và hỗ trợ
chuyển đổi (switching) giữa 2 chế độ:
- Baseline Mode: Không có TurboQuant (f32 cache)
- TurboQuant+ Mode: Có KV Cache Compression (turbo3/turbo4)

THAY ĐỔI LỚN (v2):
- Đã LOẠ BỎ HOÀN TOÀN dependency Ollama (langchain_ollama / ChatOllama
  / ENABLE_OLLAMA_FALLBACK / OLLAMA_*).
- Model giờ được tải trực tiếp về máy dưới dạng file GGUF trong thư mục
  Factaudit/models/ (ví dụ: models/Qwen3-14B-Q8_0.gguf) và được serve bởi
  server `llama-cpp-turboquant` qua giao thức OpenAI-compatible.
- Client (Factaudit) CHỈ giao tiếp với server qua REST API (ChatOpenAI);
  toàn bộ logic load/nén KV cache nằm trong server, không còn trong Factaudit.
"""

import os
from pathlib import Path
from typing import Optional, Literal
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI  # Fallback cloud (tuỳ chọn)

load_dotenv()

# config.py nằm tại: Factaudit/src/config.py -> project root = parent của src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_GGUF_FILE = "Qwen3-14B-Q8_0.gguf"


def resolve_gguf_model_path() -> Path:
    """
    Resolve đường dẫn tuyệt đối của file model GGUF trong Factaudit/models/.
    Phục vụ LOG/CẢNH BÁO (server mới là bên load file này).

    Priority:
    1. GGUF_MODEL_PATH  : đường dẫn tuyệt đối/relative chỉ định trực tiếp
    2. GGUF_MODEL_FILE  : tên file .gguf trong MODELS_DIR
    3. Scan MODELS_DIR   : file *.gguf đầu tiên
    4. Fallback          : DEFAULT_GGUF_FILE trong MODELS_DIR
    """
    explicit = os.getenv("GGUF_MODEL_PATH")
    if explicit:
        return Path(explicit).expanduser()

    model_file = os.getenv("GGUF_MODEL_FILE")
    if model_file:
        return (MODELS_DIR / model_file).resolve()

    if MODELS_DIR.is_dir():
        gguf_files = sorted(MODELS_DIR.glob("*.gguf"))
        if gguf_files:
            return gguf_files[0].resolve()

    return (MODELS_DIR / DEFAULT_GGUF_FILE).resolve()


class LLMFactory:
    """
    Factory Pattern tạo LLM instances với mode switching.
    - "baseline":   -> BASELINE_API_BASE   (port 8080, cache f32)
    - "turboquant": -> TURBOQUANT_API_BASE (port 8081, cache turbo3/turbo4)
    Cả 2 đều tạo ChatOpenAI. Không còn fallback Ollama.
    """

    def __init__(self, mode: Optional[Literal["baseline", "turboquant", "auto"]] = None):
        self._mode = self._determine_mode(mode)
        self._api_base = self._get_api_base()
        # MODEL_NAME = alias gửi trong field `model` (nên khớp --alias của server)
        self._model_name = os.getenv("MODEL_NAME", "Qwen3-14B-Q8_0")
        self._api_key = os.getenv("API_KEY", "sk-not-required")
        self._timeout = int(os.getenv("TIMEOUT", "300"))
        self._gguf_path = resolve_gguf_model_path()
        self._print_mode_info()

    def _determine_mode(self, mode: Optional[str]) -> str:
        if mode in ["baseline", "turboquant"]:
            return mode
        env_mode = os.getenv("MODE", "auto").lower()
        if env_mode in ["baseline", "turboquant"]:
            return env_mode
        use_turbo = os.getenv("USE_TURBOQUANT", "false").lower() == "true"
        return "turboquant" if use_turbo else "baseline"

    def _get_api_base(self) -> str:
        if self._mode == "turboquant":
            return os.getenv("TURBOQUANT_API_BASE", "http://localhost:8081/v1")
        return os.getenv("BASELINE_API_BASE", "http://localhost:8080/v1")

    @staticmethod
    def _truncate(text: str, width: int = 50) -> str:
        text = str(text)
        return text if len(text) <= width else "..." + text[-(width - 3):]

    def _print_mode_info(self):
        mode_display = "TurboQuant+ (KV Cache Compression)" if self._mode == "turboquant" else "Baseline (f32 cache)"
        if self._gguf_path.exists():
            gguf_status = f"found ({self._gguf_path.stat().st_size / (1024 ** 3):.1f} GB)"
        else:
            gguf_status = "NOT FOUND - server có thể không load được"

        print(f"┌" + "─" * 70 + "┐")
        print(f"│ {'LLM FACTORY INITIALIZED':^66} │")
        print(f"├" + "─" * 70 + "┤")
        print(f"│ Mode:        {self._truncate(mode_display):<50} │")
        print(f"│ API Base:    {self._truncate(self._api_base):<50} │")
        print(f"│ Model Alias: {self._truncate(self._model_name):<50} │")
        print(f"│ GGUF File:   {self._truncate(self._gguf_path.name):<50} │")
        print(f"│ GGUF Path:   {self._truncate(self._gguf_path, 50):<50} │")
        print(f"│ GGUF Status: {self._truncate(gguf_status):<50} │")
        if self._mode == "turboquant":
            ctx = os.getenv("TURBOQUANT_CONTEXT_SIZE", "32768")
            print(f"│ Max Context: {self._truncate(ctx + ' tokens (4x capacity)'):<50} │")
        else:
            ctx = os.getenv("MAX_CONTEXT_SIZE", "8192")
            print(f"│ Max Context: {self._truncate(ctx + ' tokens'):<50} │")
        print(f"└" + "─" * 70 + "┘")

    @property
    def mode(self) -> str: return self._mode
    @property
    def api_base(self) -> str: return self._api_base
    @property
    def gguf_path(self) -> Path: return self._gguf_path

    def switch_mode(self, new_mode: Literal["baseline", "turboquant"]) -> None:
        old_mode = self._mode
        self._mode = new_mode
        self._api_base = self._get_api_base()
        print(f"\n🔄 [LLMFactory] Mode switched: {old_mode.upper()} → {new_mode.upper()}")
        print(f"   New API Base: {self._api_base}\n")

    # ------------------------------------------------------------------
    def _create_base_llm(
        self,
        temperature: float,
        max_tokens: Optional[int] = None,
        format: Optional[str] = None,
    ) -> ChatOpenAI:
        if max_tokens is None:
            max_tokens = int(os.getenv("MAX_TOKENS", "4096"))

        kwargs = {
            "model": self._model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "api_key": self._api_key,
            "timeout": self._timeout,
        }
        # format giữ lại để tương thích ngược; JSON thực xử lý qua
        # with_structured_output() ở tầng agent (tránh xung đột tool calling).
        _ = format
        return ChatOpenAI(base_url=self._api_base, **kwargs)

    def _create_fallback_gemini(self, temperature: float, model: str = "gemini-2.5-flash"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY must be set in .env for Gemini fallback")
        return ChatGoogleGenerativeAI(model=model, temperature=temperature, api_key=api_key)

    def create_explorer(self) -> ChatOpenAI:
        temp = float(os.getenv("TEMPERATURE_EXPLORER", "1.0"))
        llm = self._create_base_llm(temperature=temp)
        print(f"  ✓ llm_explorer created (temp={temp})")
        return llm

    def create_judge(self, format: Optional[str] = "json") -> ChatOpenAI:
        temp = float(os.getenv("TEMPERATURE_JUDGE", "0.0"))
        llm = self._create_base_llm(temperature=temp, format=format)
        print(f"  ✓ llm_judge created (temp={temp}, format={format})")
        return llm

    def create_scorer(self) -> ChatOpenAI:
        temp = float(os.getenv("TEMPERATURE_SCORER", "0.0"))
        llm = self._create_base_llm(temperature=temp)
        print(f"  ✓ llm_scorer created (temp={temp})")
        return llm

    def create_target(self) -> ChatOpenAI:
        temp = float(os.getenv("TEMPERATURE_TARGET", "0.6"))
        llm = self._create_base_llm(temperature=temp)
        print(f"  ✓ llm_target created (temp={temp})")
        return llm

    def create_all(self) -> dict:
        print(f"\n[LLMFactory] Creating all LLM instances...")
        return {
            "explorer": self.create_explorer(),
            "judge": self.create_judge(),
            "scorer": self.create_scorer(),
            "target": self.create_target(),
        }


# ---- Global instances + helpers (giữ nguyên v1) ----
_llm_factory: Optional[LLMFactory] = None
llm_explorer = llm_judge = llm_scorer = llm_target = None

def get_factory(mode: Optional[str] = None) -> LLMFactory:
    global _llm_factory
    if _llm_factory is None or (mode is not None and _llm_factory.mode != mode):
        _llm_factory = LLMFactory(mode=mode)
    return _llm_factory

def initialize_llms(mode: Optional[str] = None) -> dict:
    global llm_explorer, llm_judge, llm_scorer, llm_target
    factory = get_factory(mode=mode)
    llms = factory.create_all()
    llm_explorer, llm_judge, llm_scorer, llm_target = (
        llms["explorer"], llms["judge"], llms["scorer"], llms["target"]
    )
    return llms

def switch_llm_mode(new_mode: Literal["baseline", "turboquant"]) -> None:
    global _llm_factory
    print(f"\n{'='*70}\n🔄 SWITCHING LLM MODE TO: {new_mode.upper()}\n{'='*70}\n")
    _llm_factory = None
    initialize_llms(mode=new_mode)
    print(f"\n✅ All LLM instances switched to {new_mode.upper()} mode!\n")

# ---- Constants ----
MAX_RETRIES = 3
MAX_WEB_CHECKS = 2
LOW_SCORE_THRESHOLD = 3.0
MAX_ITERATIONS = 3

def get_max_context_size() -> int:
    factory = get_factory()
    if factory.mode == "turboquant":
        return int(os.getenv("TURBOQUANT_CONTEXT_SIZE", "32768"))
    return int(os.getenv("MAX_CONTEXT_SIZE", "8192"))

# ---- Auto-init on import (override bởi main.py nếu có --mode) ----
_initialized = False
def ensure_initialized(mode: Optional[str] = None):
    global _initialized
    if not _initialized or (mode is not None and get_factory().mode != mode):
        initialize_llms(mode=mode)
        _initialized = True

if not _initialized:
    initialize_llms()
    _initialized = True
```

### 4.2 Bảo toàn contract với các Agent

Toàn bộ agent vẫn dùng `.with_structured_output(PydanticModel)`, nên 4 instance `llm_*` **bắt buộc** là `ChatOpenAI`. v2 đảm bảo:

| Agent / module        | LLM dùng                                             | Hợp đồng giữ nguyên                                                |
| --------------------- | ----------------------------------------------------- | ----------------------------------------------------------------------- |
| `appraiser`         | `llm_explorer` (temp 1.0), `llm_judge` (temp 0.0) | `.with_structured_output(AnalysisOutput / JudgeOutput)`               |
| `inquirer`          | `llm_judge`                                         | `.with_structured_output(InquirerOutput)`                             |
| `quality_inspector` | `llm_judge`                                         | `.with_structured_output(InspectionOutput / WebVerification)`         |
| `evaluator`         | `llm_explorer`, `llm_judge`, `llm_scorer`       | `.with_structured_output(ReferenceOutput / VoteOutput / ScoreOutput)` |
| `prober`            | `llm_explorer`                                      | `.with_structured_output(TestCase)`                                   |
| `target_model`      | `llm_target`                                        | `.with_structured_output(ReferenceOutput)`                            |

`create_judge(format="json")` **vẫn nhận tham số `format`** nhưng KHÔNG ép `response_format` (tránh xung đột với tool-calling của `with_structured_output`). Đây là cùng cách xử lý như v1 nên không phá vỡ gì.

### 4.3 File `.env` (v2)

```env
# 1. API KEYS
TAVILY_API_KEY=tvly-YOUR_TAVILY_API_KEY_HERE
GEMINI_API_KEY=                      # bỏ trống -> dùng server local

# 2. LLM MODE SELECTION
MODE=auto                            # auto | baseline | turboquant
USE_TURBOQUANT=false                 # dùng khi MODE=auto

# 3. API ENDPOINTS (llama-cpp-turboquant)
BASELINE_API_BASE=http://localhost:8080/v1
TURBOQUANT_API_BASE=http://localhost:8081/v1

# 4. MODEL GGUF (Factaudit/models/)
GGUF_MODEL_PATH=                     # đường dẫn tuyệt đối (ưu tiên cao nhất nếu set)
GGUF_MODEL_FILE=Qwen3-14B-Q8_0.gguf  # tên file trong thư mục models/
MODEL_NAME=Qwen3-14B-Q8_0            # alias gửi field "model" (khớp --alias của server)

# 5. GENERATION
API_KEY=sk-not-required
MAX_TOKENS=4096
TIMEOUT=300

# 6. CONTEXT SIZE
MAX_CONTEXT_SIZE=8192
TURBOQUANT_CONTEXT_SIZE=32768

# 7. TEMPERATURE
TEMPERATURE_EXPLORER=1.0
TEMPERATURE_JUDGE=0.0
TEMPERATURE_SCORER=0.0
TEMPERATURE_TARGET=0.6

# (KHÔNG CÒN) ENABLE_OLLAMA_FALLBACK / OLLAMA_BASE_URL / OLLAMA_MODEL
```

---

## 5. Script khởi động (Bash)

File `scripts/start_server.sh` (đã tạo sẵn trong repo). Nó resolve đường dẫn GGUF về `Factaudit/models/` và linh hoạt chạy 1 hoặc cả 2 server.

### 5.1 Cách dùng

```bash
cd Factaudit

./scripts/start_server.sh baseline     #前台, port 8080, cache f32
./scripts/start_server.sh turboquant   #前台, port 8081, cache turbo3/turbo4, ctx 32768
./scripts/start_server.sh both         # cả 2 ở nền, log -> logs/server_<mode>.log
./scripts/start_server.sh --help       # trợ giúp
```

### 5.2 Override biến môi trường

Script có giá trị mặc định hợp lý, nhưng bạn có thể override:

```bash
# Dùng bản build cũ (binary tên "server")
export LLAMA_SERVER_BIN=/path/to/llama-cpp-turboquant/build/bin/server

# Đổi file GGUF
export GGUF_MODEL=/data/models/Qwen3-14B-Q8_0.gguf

# Chạy cả 2 với GPU layers lớn hơn
GPU_LAYERS=99 ./scripts/start_server.sh both

# Đổi port / context
BASELINE_PORT=8000 TURBOQUANT_PORT=8001 ./scripts/start_server.sh both
```

### 5.3 Logic chính của script

| Bước         | Mô tả                                                                                                                                       |
| -------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| Resolve path   | `SCRIPT_DIR/..` → `PROJECT_ROOT`, `models/` → `GGUF_MODEL`, `MODEL_ALIAS` = tên GGUF bỏ đuôi                                  |
| Preflight      | Kiểm tra binary server & file GGUF tồn tại, báo lỗi rõ ràng nếu thiếu                                                                |
| `baseline`   | `--port 8080 --cache-type-k/v f32 --ctx-size 8192 --batch-size 512/128`                                                                     |
| `turboquant` | `--port 8081 --cache-type-k turbo3 --cache-type-v turbo4 --ctx-size 32768 --batch-size 1024/256 --turbo-quant-bits 4 --turbo-group-size 64` |
| `both`       | Chạy cả 2 ở background, ghi`logs/server_*.log` + `logs/server_*.pid`, in hướng dẫn health check & cách kill                        |

---

## 6. Cách sử dụng (Usage Examples)

### 6.1 Quy trình đầy đủ (v2)

```bash
# 1. Tải GGUF về Factaudit/models/
cd Factaudit && mkdir -p models
# (đặt Qwen3-14B-Q8_0.gguf vào models/)

# 2. Build llama-cpp-turboquant (1 lần)
cd ../llama-cpp-turboquant
mkdir build && cd build && cmake .. -DLLAMA_CUBLAS=ON -DLLAMA_TURBOQUANT=ON
cmake --build . --config Release -j$(nproc)

# 3. Cài đặt Python deps
cd ../../Factaudit
pip install -r src/requirements.txt

# 4. Cấu hình .env (đặc biệt GGUF_MODEL_FILE, MODEL_NAME, MODE)

# 5. Khởi động server (mở terminal riêng)
./scripts/start_server.sh both        # hoặc baseline / turboquant

# 6. Chạy FACT-AUDIT
python src/main.py --mode auto        # hoặc --mode baseline / --mode turboquant
```

### 6.2 Baseline Mode

```bash
# Terminal 1
./scripts/start_server.sh baseline

# Terminal 2
python src/main.py --mode baseline
```

### 6.3 TurboQuant+ Mode

```bash
# Terminal 1
./scripts/start_server.sh turboquant

# Terminal 2
python src/main.py --mode turboquant
```

### 6.4 Auto Mode (từ `.env`)

```bash
# .env: USE_TURBOQUANT=true  -> khởi động server turboquant (8081)
# .env: USE_TURBOQUANT=false -> khởi động server baseline (8080)
./scripts/start_server.sh $(python -c "import os;from dotenv import load_dotenv;load_dotenv();print('turboquant' if os.getenv('USE_TURBOQUANT','false').lower()=='true' else 'baseline')")

python src/main.py --mode auto
```

### 6.5 Python API (mode switching runtime)

```python
from config import switch_llm_mode, llm_judge, get_factory

# Chạy baseline trước, sau đó chuyển sang turboquant cho context dài:
switch_llm_mode("turboquant")
print("Now using:", get_factory().api_base)   # http://localhost:8081/v1
```

---

## 7. So sánh Hiệu năng (Performance Comparison)

> Số liệu mang tính tham khảo (model Qwen3-14B Q8_0 trên GPU 12GB). Khuyến nghị đo lại trên máy thật bằng `--metrics` của server.

| Metric                | Baseline (f32 cache) | TurboQuant+ (turbo3/4) | Improvement                |
| --------------------- | -------------------- | ---------------------- | -------------------------- |
| VRAM (8K ctx)         | ~8.2 GB              | ~3.1 GB                | **~62% giảm**       |
| Max Context           | 8,192 tokens         | 32,768 tokens          | **4× dung lượng** |
| Tốc độ inference   | ~45 t/s              | ~42 t/s                | ~7% chậm hơn             |
| Quality (perplexity)  | 7.82                 | 7.89                   | +0.9% (tối thiểu)        |
| Long-context accuracy | —                   | +2.3%                  | Tốt hơn                  |

| Use case                     | Mode đề xuất | Lý do               |
| ---------------------------- | --------------- | -------------------- |
| Claim ngắn (<2K tokens)     | Baseline        | Nhanh, đơn giản   |
| Tài liệu dài (>8K tokens) | TurboQuant+     | Xử lý context lớn |
| Batch processing             | Baseline        | Throughput cao nhất |
| GPU hạn chế VRAM           | TurboQuant+     | Tiết kiệm VRAM     |

---

## 8. Xử lý sự cố (Troubleshooting)

### 8.1 Các lỗi thường gặp (v2)

| Issue                             | Triệu chứng                                          | Cách xử lý                                                                              |
| --------------------------------- | ------------------------------------------------------ | ------------------------------------------------------------------------------------------ |
| Server binary không thấy        | Script báo`[ERROR] Không tìm thấy server binary` | Build trước, hoặc`export LLAMA_SERVER_BIN=/path/...` (đặt `server` nếu bản cũ) |
| File GGUF thiếu                  | Script báo`Không tìm thấy file GGUF`             | Tải GGUF vào`Factaudit/models/` hoặc `export GGUF_MODEL=/path/...`                  |
| `field "model" not found` / 404 | Client gọi API bị server từ chối                   | Đặt`MODEL_NAME` (`.env`) = `--alias` của server                                   |
| Timeout khi invoke                | LLM gọi lâu / lỗi kết nối                         | Kiểm tra server chạy, tăng`TIMEOUT`, xem `logs/server_*.log`                        |
| VRAM OOM                          | CUDA out of memory                                     | Giảm`--ctx-size` hoặc `--n-gpu-layers`, hoặc dùng TurboQuant                       |
| Port bị chiếm                   | `address already in use`                             | Đổi`BASELINE_PORT`/`TURBOQUANT_PORT`, hoặc `kill $(cat logs/server_*.pid)`        |
| Import`langchain_ollama` lỗi   | (nếu còn code cũ)                                   | Đảm bảo đã xoá import trong`config.py` và gỡ `langchain-ollama`                |

### 8.2 Lệnh debug

```bash
# Trạng thái 2 server
curl http://localhost:8080/v1/models
curl http://localhost:8081/v1/models

# Theo dõi VRAM
nvidia-smi -l 1

# Xem log server (khi chạy both)
tail -f logs/server_baseline.log
tail -f logs/server_turboquant.log

# Dừng server chạy nền
kill $(cat logs/server_*.pid)

# Kiểm tra Factaudit có còn nhắc Ollama không (phải rỗng)
grep -rin "ollama" src/ requirements.txt .env
```

---

## 9. Appendix

### A. `requirements.txt` (v2)

```txt
# Core
langgraph>=0.0.26
langchain-core>=0.1.52

# LLM client (CHÍNH - thay thế Ollama)
langchain-openai>=0.1.20

# Cloud fallback tuỳ chọn
langchain-google-genai>=1.0.1

# Tools & search
langchain-tavily>=0.1.0

# Utilities
pydantic>=2.0.0
python-dotenv>=1.0.0
typing-extensions>=4.6.0
httpx>=0.27.0

# (ĐÃ XOÁ) langchain-ollama>=0.1.0
```

### B. Quick Start (v2)

```bash
# 1. Clone repos
git clone https://github.com/your-org/Factaudit.git
git clone https://github.com/your-org/llama-cpp-turboquant.git

# 2. Tải GGUF vào Factaudit/models/
cd Factaudit && mkdir -p models
# -> đặt Qwen3-14B-Q8_0.gguf vào models/

# 3. Build server
cd ../llama-cpp-turboquant
mkdir build && cd build
cmake .. -DLLAMA_CUBLAS=ON -DLLAMA_TURBOQUANT=ON
cmake --build . --config Release -j$(nproc)

# 4. Cài deps Python
cd ../../Factaudit
pip install -r src/requirements.txt
cp .env.example .env   # (nếu có) rồi chỉnh GGUF_MODEL_FILE, MODEL_NAME, MODE

# 5. Khởi động server
./scripts/start_server.sh both          # hoặc baseline / turboquant

# 6. Chạy hệ thống
python src/main.py --mode turboquant    # hoặc baseline / auto
```

### C. Ánh xạ biến env v1 → v2

| v1                           | v2                                                                     |
| ---------------------------- | ---------------------------------------------------------------------- |
| `OLLAMA_BASE_URL`          | *(xoá)* — giờ là `BASELINE_API_BASE` / `TURBOQUANT_API_BASE` |
| `OLLAMA_MODEL`             | *(xoá)* — giờ là `GGUF_MODEL_FILE` + `MODEL_NAME`            |
| `ENABLE_OLLAMA_FALLBACK`   | *(xoá)*                                                             |
| `MODEL_NAME` (tên Ollama) | `MODEL_NAME` = alias GGUF (vd `Qwen3-14B-Q8_0`)                    |
| —*(mới)*                 | `GGUF_MODEL_PATH`, `GGUF_MODEL_FILE`                               |

---

**Document Version:** 2.0
**Last Updated:** 2026-06-28
**Thay đổi nổi bật:** Loại bỏ hoàn toàn Ollama; model GGUF cục bộ trong `Factaudit/models/`; client chỉ dùng `ChatOpenAI`; thêm script `scripts/start_server.sh`.
