#!/usr/bin/env bash
# =====================================================================
#  start_server.sh  --  Khởi chạy llama-cpp-turboquant server (v3 - Dual Model)
# =====================================================================
#  Mục đích: khởi chạy tới 4 server = 2 MODEL ĐỘC LẬP × 2 MODE (baseline/turbo).
#
#    Model A (5 agent):  Qwen3-32B-Q8_0.gguf
#      - baseline   -> port 8080 (cache f32)
#      - turboquant -> port 8081 (cache turbo3/turbo4)
#    Model B (Target):   Qwen3-14B-Q8_0.gguf
#      - baseline   -> port 8082 (cache f32)
#      - turboquant -> port 8083 (cache turbo3/turbo4)
#
#  Mode-switching trong Factaudit là TOÀN CỤC: --mode baseline sẽ dùng
#  A-baseline + B-baseline; --mode turboquant sẽ dùng A-turbo + B-turbo.
#
#  Usage:
#     ./scripts/start_server.sh                 # = both (cả 4 server nền)
#     ./scripts/start_server.sh baseline        # 2 server baseline (8080 + 8082)
#     ./scripts/start_server.sh turboquant      # 2 server turboquant (8081 + 8083)
#     ./scripts/start_server.sh model-a         # cả 2 server của Model A
#     ./scripts/start_server.sh model-b         # cả 2 server của Model B
#     ./scripts/start_server.sh a-turbo         # chỉ 1 server (dev VRAM thấp)
#
#  Override cấu hình qua biến môi trường (có giá trị mặc định):
#     LLAMA_CPP_DIR         thư mục repo llama-cpp-turboquant
#     LLAMA_SERVER_BIN      đường dẫn binary server (mặc định <DIR>/build/bin/llama-server)
#     MODEL_A_GGUF/ALIAS    file GGUF + alias của Model A
#     MODEL_B_GGUF/ALIAS    file GGUF + alias của Model B
#     MODEL_A_*_PORT/MODEL_B_*_PORT   port của từng server
#     GPU_LAYERS / THREADS / HOST / *_CTX / TURBO_* ...
#
#  Lưu ý: bản build cũ của llama.cpp đặt tên binary là `server` (đường dẫn
#         build/bin/server). Bản mới dùng `llama-server`. Nếu báo "command
#         not found", hãy set:  export LLAMA_SERVER_BIN=/path/to/server
# =====================================================================
set -euo pipefail

# ---------------------------------------------------------------------
# Resolve đường dẫn (script nằm trong Factaudit/scripts/)
# ---------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODELS_DIR="$PROJECT_ROOT/models"

# Repo llama-cpp-turboquant (mặc định là anh em với Factaudit)
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$PROJECT_ROOT/../llama-cpp-turboquant}"
# Binary server: bản mới là llama-server, bản cũ là server
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-$LLAMA_CPP_DIR/build/bin/llama-server}"

# ---- File GGUF + alias từng model ----
MODEL_A_GGUF="${MODEL_A_GGUF:-$MODELS_DIR/Qwen3-32B-Q8_0.gguf}"
MODEL_A_ALIAS="${MODEL_A_ALIAS:-$(basename "$MODEL_A_GGUF" .gguf)}"
MODEL_B_GGUF="${MODEL_B_GGUF:-$MODELS_DIR/Qwen3-14B-Q8_0.gguf}"
MODEL_B_ALIAS="${MODEL_B_ALIAS:-$(basename "$MODEL_B_GGUF" .gguf)}"

# ---- Tham số server chung ----
THREADS="${THREADS:-8}"
GPU_LAYERS="${GPU_LAYERS:-35}"
HOST="${HOST:-0.0.0.0}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs}"

# ---- Port từng server ----
MODEL_A_BASELINE_PORT="${MODEL_A_BASELINE_PORT:-8080}"
MODEL_A_TURBO_PORT="${MODEL_A_TURBO_PORT:-8081}"
MODEL_B_BASELINE_PORT="${MODEL_B_BASELINE_PORT:-8082}"
MODEL_B_TURBO_PORT="${MODEL_B_TURBO_PORT:-8083}"

# ---- Context size từng mode ----
BASELINE_CTX="${BASELINE_CTX:-8192}"
TURBOQUANT_CTX="${TURBOQUANT_CTX:-32768}"

# ---- TurboQuant specifics (chung cho cả 2 model) ----
TURBO_K="${TURBO_K:-turbo3}"
TURBO_V="${TURBO_V:-turbo4}"
TURBO_QUANT_BITS="${TURBO_QUANT_BITS:-4}"
TURBO_GROUP_SIZE="${TURBO_GROUP_SIZE:-64}"

mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
c_red()    { printf '\033[31m%s\033[0m' "$*"; }
c_green()  { printf '\033[32m%s\033[0m' "$*"; }
c_cyan()   { printf '\033[36m%s\033[0m' "$*"; }
c_yellow() { printf '\033[33m%s\033[0m' "$*"; }

banner() {
  echo
  echo "$(c_cyan '==================================================')"
  echo "  $1"
  echo "$(c_cyan '==================================================')"
}

# Kiểm tra binary server tồn tại.
check_binary() {
  if [[ ! -x "$LLAMA_SERVER_BIN" && ! -f "$LLAMA_SERVER_BIN" ]]; then
    echo "$(c_red '[ERROR]') Không tìm thấy server binary:"
    echo "    $LLAMA_SERVER_BIN"
    echo "  -> Build llama-cpp-turboquant trước, hoặc set LLAMA_SERVER_BIN."
    return 1
  fi
  return 0
}

# ---------------------------------------------------------------------
# Khởi chạy 1 server. Background nếu $2 == "bg".
#   $1 = tag (a_baseline|a_turbo|b_baseline|b_turbo)
#   $2 = mode chạy (bg|fg)
#   $3 = file GGUF
#   $4 = alias
#   $5 = port
#   $6 = cache type K (baseline=f32, turbo=turbo3)
#   $7 = cache type V (baseline=f32, turbo=turbo4)
#   $8 = context size
#   $9 = "turbo" để bật thêm --turbo-quant-* flags, ngược lại "baseline"
# ---------------------------------------------------------------------
run_server() {
  local tag="$1"
  local bg="${2:-fg}"
  local gguf="$3"
  local alias="$4"
  local port="$5"
  local cache_k="$6"
  local cache_v="$7"
  local ctx="$8"
  local qmode="${9:-baseline}"
  local log_file="$LOG_DIR/server_${tag}.log"
  local pid_file="$LOG_DIR/server_${tag}.pid"

  # Hard-exit nếu GGUF của server đang yêu cầu mà không tồn tại
  if [[ ! -f "$gguf" ]]; then
    echo "$(c_red '[ERROR]') [$tag] Không tìm thấy file GGUF: $gguf"
    echo "  -> Tải model về Factaudit/models/ hoặc set biến *_GGUF tương ứng."
    return 1
  fi

  # Chọn batch size theo mode (turbo dùng batch lớn hơn)
  local batch ubatch extra=()
  if [[ "$qmode" == "turbo" ]]; then
    batch=1024; ubatch=256
    extra+=(--turbo-quant-bits "$TURBO_QUANT_BITS" --turbo-group-size "$TURBO_GROUP_SIZE")
  else
    batch=512; ubatch=128
  fi

  local cmd=(
    "$LLAMA_SERVER_BIN"
      --host "$HOST"
      --port "$port"
      --model "$gguf"
      --alias "$alias"
      --threads "$THREADS"
      --n-gpu-layers "$GPU_LAYERS"
      --ctx-size "$ctx"
      --batch-size "$batch"
      --ubatch-size "$ubatch"
      --cache-type-k "$cache_k"
      --cache-type-v "$cache_v"
      --metrics
      --log-format text
  )
  cmd+=("${extra[@]}")

  if [[ "$bg" == "bg" ]]; then
    echo "$(c_green "[start]") $tag -> background"
    echo "    alias : $alias"
    echo "    port  : $port"
    echo "    gguf  : $gguf"
    echo "    cache : K=$cache_k V=$cache_v  ctx=$ctx"
    echo "    log   : $log_file"
    "${cmd[@]}" > "$log_file" 2>&1 &
    echo $! > "$pid_file"
    echo "    pid   : $(cat "$pid_file")"
  else
    banner "Server: $tag  ($alias @ $port, $qmode)  (Ctrl+C để dừng)"
    echo "  cmd: ${cmd[*]}"
    echo "  log: $log_file  (tee song song terminal)"
    echo
    "${cmd[@]}" 2>&1 | tee "$log_file"
  fi
}

# ---- Builder tiện lợi: tránh nhầm thứ tự tham số khi gọi run_server ----
# Cú pháp gọi thống nhất: <tag> <bg|fg> <gguf> <alias> <port> <ctx> <baseline|turbo>
_launch() {
  local tag="$1" bg="$2" gguf="$3" alias="$4" port="$5" ctx="$6" qmode="$7"
  if [[ "$qmode" == "turbo" ]]; then
    run_server "$tag" "$bg" "$gguf" "$alias" "$port" "$TURBO_K" "$TURBO_V" "$ctx" turbo
  else
    run_server "$tag" "$bg" "$gguf" "$alias" "$port" f32 f32 "$ctx" baseline
  fi
}

# ---------------------------------------------------------------------
# Định nghĩa 4 server (dùng bởi các action bên dưới)
#   args: <bg|fg>
# ---------------------------------------------------------------------
start_a_baseline() { _launch a_baseline "$1" "$MODEL_A_GGUF" "$MODEL_A_ALIAS" "$MODEL_A_BASELINE_PORT" "$BASELINE_CTX" baseline; }
start_a_turbo()    { _launch a_turbo    "$1" "$MODEL_A_GGUF" "$MODEL_A_ALIAS" "$MODEL_A_TURBO_PORT"    "$TURBOQUANT_CTX" turbo; }
start_b_baseline() { _launch b_baseline "$1" "$MODEL_B_GGUF" "$MODEL_B_ALIAS" "$MODEL_B_BASELINE_PORT" "$BASELINE_CTX" baseline; }
start_b_turbo()    { _launch b_turbo    "$1" "$MODEL_B_GGUF" "$MODEL_B_ALIAS" "$MODEL_B_TURBO_PORT"    "$TURBOQUANT_CTX" turbo; }

# Cảnh báo VRAM tổng khi chạy nhiều server cùng lúc (ước lượng thô).
vram_warn() {
  local n="$1"
  if [[ "$n" -ge 4 ]]; then
    echo "$(c_yellow '[WARN]') Đang khởi động $n server cùng lúc."
    echo "   Ước lượng VRAM (offload đầy đủ):"
    echo "     Model A (32B Q8) x2 ≈ 66 GB   |   Model B (14B Q8) x2 ≈ 30 GB"
    echo "     Tổng ≈ 96 GB+  -> chỉ khả thi trên rig lớn (2x48GB / 4x24GB / H100)."
    echo "   Nếu thiếu VRAM, hạ GPU_LAYERS hoặc chạy action granular (vd: model-a, a-turbo)."
    echo
  fi
}

# In health-check + hướng dẫn dừng cho cả 4.
print_both_epilog() {
  echo
  echo "$(c_green '[ok]') Đã khởi động các server."
  echo "  Model A baseline : http://localhost:$MODEL_A_BASELINE_PORT/v1/models  ($MODEL_A_ALIAS)"
  echo "  Model A turbo    : http://localhost:$MODEL_A_TURBO_PORT/v1/models     ($MODEL_A_ALIAS)"
  echo "  Model B baseline : http://localhost:$MODEL_B_BASELINE_PORT/v1/models  ($MODEL_B_ALIAS)"
  echo "  Model B turbo    : http://localhost:$MODEL_B_TURBO_PORT/v1/models     ($MODEL_B_ALIAS)"
  echo
  echo "  Xem log : tail -f $LOG_DIR/server_{a_baseline,a_turbo,b_baseline,b_turbo}.log"
  echo "  Dừng    : kill \$(cat $LOG_DIR/server_*.pid)"
  echo
  echo "  Health check:"
  echo "    curl -s http://localhost:$MODEL_A_BASELINE_PORT/health && echo '  a_baseline ok'"
  echo "    curl -s http://localhost:$MODEL_A_TURBO_PORT/health    && echo '  a_turbo ok'"
  echo "    curl -s http://localhost:$MODEL_B_BASELINE_PORT/health && echo '  b_baseline ok'"
  echo "    curl -s http://localhost:$MODEL_B_TURBO_PORT/health    && echo '  b_turbo ok'"
}

usage() {
  # In các dòng comment ở đầu file (bỏ qua shebang + dòng trống) cho tới khi
  # gặp dòng code đầu tiên — robust với thay đổi độ dài header.
  awk 'NR==1{next} /^[[:space:]]*$/{next} /^#/{sub(/^#[[:space:]]?/,""); print; next} {exit}' \
    "${BASH_SOURCE[0]}"
  exit 1
}

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
check_binary
ACTION="${1:-both}"

case "$ACTION" in
  baseline)
    banner "Khởi động 2 server BASELINE (A:8080 + B:8082) ở nền"
    vram_warn 2
    start_a_baseline bg
    start_b_baseline bg
    ;;
  turboquant|turbo)
    banner "Khởi động 2 server TURBOQUANT (A:8081 + B:8083) ở nền"
    vram_warn 2
    start_a_turbo bg
    start_b_turbo bg
    ;;
  model-a)
    banner "Khởi động cả 2 server MODEL A (32B) ở nền"
    vram_warn 2
    start_a_baseline bg
    start_a_turbo bg
    ;;
  model-b)
    banner "Khởi động cả 2 server MODEL B (14B) ở nền"
    vram_warn 2
    start_b_baseline bg
    start_b_turbo bg
    ;;
  both)
    banner "Khởi động CẢ 4 server ở nền (background)"
    vram_warn 4
    start_a_baseline bg
    start_a_turbo bg
    start_b_baseline bg
    start_b_turbo bg
    print_both_epilog
    ;;
  a-baseline) start_a_baseline fg ;;
  a-turbo)    start_a_turbo fg ;;
  b-baseline) start_b_baseline fg ;;
  b-turbo)    start_b_turbo fg ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "$(c_red "[ERROR]") Action không hợp lệ: $ACTION"
    echo "  Hợp lệ: baseline | turboquant | both | model-a | model-b | a-baseline | a-turbo | b-baseline | b-turbo"
    usage
    ;;
esac
