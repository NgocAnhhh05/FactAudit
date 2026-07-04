# FACT-AUDIT: Multi-Agent System for Fact-Checking & LLM Vulnerability Assessment

FACT-AUDIT là một framework kiểm toán tự động sử dụng kiến trúc Multi-Agent (LangGraph) để đánh giá lỗ hổng, kiểm tra tính xác thực dữ liệu và chấm điểm các mô hình ngôn ngữ lớn (Target LLM). Hệ thống hoạt động theo mô hình Client-Server tách biệt, sử dụng `llama-cpp-turboquant` làm backend inference hiệu năng cao và loại bỏ hoàn toàn dependency vào Ollama.

---

## 🏗️ Kiến Trúc Hệ Thống

Hệ thống điều phối luồng xử lý thông qua 5 Agents cốt lõi kết hợp với mô hình Target LLM:
1. **Appraiser Agent**: Phân tích ngữ cảnh, lập kịch bản kiểm tra độc lập.
2. **Prober Agent**: Tạo các test-case thử nghiệm cấu trúc/bảo mật.
3. **Inquirer Agent**: Thực thi các câu hỏi, giám sát phản hồi.
4. **Quality Inspector Agent**: Xác minh chéo thông tin bằng cách kết hợp với công cụ tìm kiếm bên ngoài (Tavily Search).
5. **Evaluator Agent**: Đánh giá câu trả lời, bỏ phiếu chấm điểm và trích xuất số liệu Metrics (Grade, IMR, JFR).


```

┌────────────────────────────────────────────────────────┐
│               FACT-AUDIT (Client Layer)                │
│  ┌───────────────────────┐    ┌──────────────────────┐ │
│  │ LangGraph Orchestrator│ ──►│  LLMFactory (ChatOpenAI)│ │
│  └───────────────────────┘    └──────────────────────┘ │
└───────────────────────────┬────────────────────────────┘
│ OpenAI API REST Protocol
┌───────────────────────────▼────────────────────────────┐
│         llama-cpp-turboquant (Server Layer)            │
│  ┌──────────────────────────────────────────────────┐  │
│  │   [Model A: 5 Agents - Qwen2.5-32B-Instruct]     │  │
│  │   - Baseline Mode   : Port 8080 (Cache f32)      │  │
│  │   - TurboQuant Mode : Port 8081 (Cache turbo3/4) │  │
│  └──────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────┐  │
│  │   [Model B: Target LLM - Qwen2.5-14B-Instruct]   │  │
│  │   - Baseline Mode   : Port 8082 (Cache f32)      │  │
│  │   - TurboQuant Mode : Port 8083 (Cache turbo3/4) │  │
│  └──────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────┘

```

---

## ⚙️ Cấu Hình Hệ Thống (`.env`)

Tạo file `.env` tại thư mục gốc của dự án với nội dung cấu hình như sau:

```env
# 1. API KEYS
TAVILY_API_KEY=tvly-YOUR_KEY_HERE

# 2. LLM MODE SELECTION
MODE=auto                            # auto | baseline | turboquant
USE_TURBOQUANT=false                 # cờ kích hoạt nếu MODE=auto

# 3. ENDPOINTS CONFIGURATION
MODEL_A_BASELINE_API_BASE=[http://127.0.0.1:8080/v1](http://127.0.0.1:8080/v1)
MODEL_A_TURBOQUANT_API_BASE=[http://127.0.0.1:8081/v1](http://127.0.0.1:8081/v1)

MODEL_B_BASELINE_API_BASE=[http://127.0.0.1:8082/v1](http://127.0.0.1:8082/v1)
MODEL_B_TURBOQUANT_API_BASE=[http://127.0.0.1:8083/v1](http://127.0.0.1:8083/v1)

# 4. LOCAL MODELS DEFINITION
GGUF_MODEL_FILE=Qwen3-14B-Q4_K_M.gguf
MODEL_NAME=Qwen3-14B-Q4_K_M
API_KEY=sk-not-required

# 5. CONTEXT SIZE
MAX_CONTEXT_SIZE=8192
TURBOQUANT_CONTEXT_SIZE=32768

```

---

## 🚀 Hướng Dẫn Kích Hoạt Thực Nghiệm

### Bước 1: Chuẩn bị Thư mục và Tải file Mô hình GGUF



Tải phiên bản lượng hóa Single File (Q4_K_M) trực tiếp về thư mục lưu trữ cục bộ:

```bash
cd Factaudit
mkdir -p models && cd models

# Tải Model A (Bộ não 5 Agents)
wget -O Qwen2.5-32B-Instruct-Q4_K_M.gguf [https://huggingface.co/bartowski/Qwen2.5-32B-Instruct-GGUF/resolve/main/Qwen2.5-32B-Instruct-Q4_K_M.gguf](https://huggingface.co/bartowski/Qwen2.5-32B-Instruct-GGUF/resolve/main/Qwen2.5-32B-Instruct-Q4_K_M.gguf)

# Tải Model B (Target LLM Thí nghiệm)
wget -O Qwen3-14B-Q4_K_M.gguf [https://huggingface.co/bartowski/Qwen2.5-14B-Instruct-GGUF/resolve/main/Qwen2.5-14B-Instruct-Q4_K_M.gguf](https://huggingface.co/bartowski/Qwen2.5-14B-Instruct-GGUF/resolve/main/Qwen2.5-14B-Instruct-Q4_K_M.gguf)

```

### Bước 2: Cài đặt Môi trường Python (Client)



```bash
cd ~/Factaudit
python3 -m venv venv
source venv/bin/activate
pip install -r src/requirements.txt
pip install pandas filelock

```

### Bước 3: Biên dịch Backend C++ `llama-cpp-turboquant`

```bash
cd ~/llama-cpp-turboquant
mkdir -p build && cd build
cmake .. -DGGML_CUDA=ON -DLLAMA_TURBOQUANT=ON
cmake --build . --config Release -j$(nproc)

```

### Bước 4: Khởi động hệ thống máy chủ Inference (Thủ công theo cặp Port)

#### Chế độ A: Chạy thực nghiệm BASELINE Mode (Port 8080 + 8082)



* **Terminal 1 (Model A Baseline Server):**
```bash
cd ~/llama-cpp-turboquant/build
./bin/llama-server --host 0.0.0.0 --port 8080 --model "/home/ngocanhdoan1092005/Factaudit/models/Qwen2.5-32B-Instruct-Q4_K_M.gguf" --alias Qwen2.5-32B-Instruct-Q4_K_M --threads 8 --n-gpu-layers 45 --ctx-size 8192 --cache-type-k f32 --cache-type-v f32 --metrics

```


* **Terminal 2 (Model B Baseline Server - Chạy ngầm trong `tmux new -s target_baseline`):**
```bash
cd ~/llama-cpp-turboquant/build
./bin/llama-server --host 0.0.0.0 --port 8082 --model "/home/ngocanhdoan1092005/Factaudit/models/Qwen3-14B-Q4_K_M.gguf" --alias Qwen3-14B-Q4_K_M --threads 8 --n-gpu-layers 15 --ctx-size 8192 --cache-type-k f32 --cache-type-v f32 --metrics

```



#### Chế độ B: Chạy thực nghiệm TURBOQUANT Mode (Port 8081 + 8083)



* **Terminal 1 (Model A TurboQuant Server):**
```bash
cd ~/llama-cpp-turboquant/build
./bin/llama-server --host 0.0.0.0 --port 8081 --model "/home/ngocanhdoan1092005/Factaudit/models/Qwen2.5-32B-Instruct-Q4_K_M.gguf" --alias Qwen2.5-32B-Instruct-Q4_K_M --threads 8 --n-gpu-layers 45 --ctx-size 32768 --cache-type-k turbo3 --cache-type-v turbo4 --metrics

```


* **Terminal 2 (Model B TurboQuant Server - Chạy ngầm trong `tmux new -s target_turboquant`):**
```bash
cd ~/llama-cpp-turboquant/build
./bin/llama-server --host 0.0.0.0 --port 8083 --model "/home/ngocanhdoan1092005/Factaudit/models/Qwen3-14B-Q4_K_M.gguf" --alias Qwen3-14B-Q4_K_M --threads 8 --n-gpu-layers 15 --ctx-size 32768 --cache-type-k turbo3 --cache-type-v turbo4 --metrics

```



---

## 🚀 Thực Thi Đồ Thị Kiểm Toán (Execution)

Mở một cửa sổ Terminal độc lập (Khuyến nghị đóng gói trong `tmux new -s factaudit_main`), thiết lập khóa API Search và thực thi mã nguồn điều phối chính:

```bash
cd ~/Factaudit
source venv/bin/activate
export TAVILY_API_KEY="tvly-YOUR_KEY_HERE"

# Chạy thực nghiệm luồng Baseline
python src/main.py --mode baseline

# Hoặc chạy thực nghiệm luồng TurboQuant
python src/main.py --mode turboquant

```

---

## 📊 Chỉ Số Đánh Giá & Thu Thập Số Liệu

Hệ thống hỗ trợ đo đạc và so sánh tự động thông qua hai phương thức:

1. **Phần cứng (Hardware Profiling)**: Sử dụng lệnh `nvidia-smi -l 1` để theo dõi lượng VRAM đỉnh (Peak VRAM) tiêu thụ và đối chứng khả năng tiết kiệm bộ nhớ khi nén KV Cache.
2. **Hiệu năng xử lý (Inference Profile)**: Trích xuất chỉ số từ log server thông qua tham số `--metrics` bao gồm:
* `prompt eval time` (tốc độ xử lý câu hỏi đầu vào - tokens/s).


* `eval time` (tốc độ sinh từ mã hóa đầu ra - tokens/s).


* Kích thước lưu trữ KV Cache thực tế cho mỗi phiên làm việc (`MiB/prompt`).


```