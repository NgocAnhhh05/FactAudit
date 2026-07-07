"""
===============================================
FACT-AUDIT Configuration Module  (v3 - Dual Model)
===============================================
Module này quản lý việc khởi tạo LLM instances và hỗ trợ
chuyển đổi (switching) giữa 2 chế độ:
- Baseline Mode: Không có TurboQuant (f32 cache)
- TurboQuant+ Mode: Có KV Cache Compression (turbo3/turbo4)

THAY ĐỔI LỚN (v3 - Dual Model):
- Hệ thống giờ dùng 2 MODEL ĐỘC LẬP, mỗi model đều có đủ 2 chế độ
  (baseline + turboquant) => cần 4 server llama-cpp-turboquant:
    * Model A (5 agent):  Qwen3-32B-Q8_0.gguf  -> baseline:8080 / turboquant:8081
    * Model B (Target):   Qwen3-14B-Q8_0.gguf  -> baseline:8082 / turboquant:8083
- 5 agent (Appraiser, Inquirer, Quality Inspector, Evaluator, Prober)
  luôn dùng Model A (explorer/judge/scorer). Target Model dùng Model B
  (llm_target) — hoàn toàn độc lập với các agent.
- MODE-SWITCHING LÀ TOÀN CỤC: một tham số --mode (baseline|turboquant)
  flip cả Model A và Model B cùng lúc. Không có mode riêng từng model.

(Trừ thay đổi trên, phần còn lại giữ nguyên như v2: đã LOẠI BỎ hoàn toàn
dependency Ollama. Model được tải về máy dưới dạng file GGUF trong thư mục
Factaudit/models/ và được serve bởi server llama-cpp-turboquant qua giao
thức OpenAI-compatible. Client (Factaudit) chỉ giao tiếp qua REST API.)

Cấu trúc:
- LLMFactory: Factory pattern tạo LLM instances dựa trên mode (baseline|turboquant)
  với 2 model config (model_a cho agent, model_b cho target)
- Global Constants: Các hằng số sử dụng trong hệ thống
"""

import os
from pathlib import Path
from typing import Optional, Literal
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI  # Fallback cloud (tuỳ chọn)

# ==========================================
# LOAD ENVIRONMENT VARIABLES
# ==========================================
load_dotenv()

# ==========================================
# PATH CONSTANTS
# ==========================================
# config.py nằm tại: Factaudit/src/config.py
#   -> project root = parent của src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Thư mục chứa các file model GGUF tải trực tiếp về máy
MODELS_DIR = PROJECT_ROOT / "models"


def resolve_gguf_model_path(gguf_file: Optional[str] = None) -> Path:
    """
    Resolve đường dẫn tuyệt đối của file model GGUF trong Factaudit/models/.

    Phục vụ mục đích LOG/CẢNH BÁO cho người dùng biết file nào đang được
    server sử dụng. Client không trực tiếp load file này (server đã load rồi).

    Priority (cao -> thấp):
    1. gguf_file arg     : tên file .gguf (thường được truyền từ config dict
                           của từng model). Nếu được truyền, ưu tiên cao nhất.
    2. GGUF_MODEL_PATH   : đường dẫn tuyệt đối/relative chỉ định trực tiếp
    3. GGUF_MODEL_FILE   : tên file .gguf nằm trong MODELS_DIR
    4. Scan MODELS_DIR    : file *.gguf đầu tiên tìm được (sắp xếp theo tên)
    5. Fallback           : DEFAULT_GGUF_FILE trong MODELS_DIR

    Returns:
        Path tới file GGUF (path luôn trả về, dù file có thể chưa tồn tại).
    """
    # 1. Tên file truyền vào trực tiếp (ưu tiên cao nhất) — dùng cho model_a/model_b
    if gguf_file:
        return (MODELS_DIR / gguf_file).resolve()

    # 2. Đường dẫn tuyệt đối (env override)
    explicit = os.getenv("GGUF_MODEL_PATH")
    if explicit:
        return Path(explicit).expanduser()

    # 3. Tên file nằm trong thư mục models/
    model_file = os.getenv("GGUF_MODEL_FILE")
    if model_file:
        return (MODELS_DIR / model_file).resolve()

    # 4. Scan thư mục models/ lấy file .gguf đầu tiên
    if MODELS_DIR.is_dir():
        gguf_files = sorted(MODELS_DIR.glob("*.gguf"))
        if gguf_files:
            return gguf_files[0].resolve()

    # 5. Fallback mặc định
    return (MODELS_DIR / "Qwen3-14B-Q8_0.gguf").resolve()


# ==========================================
# MODEL ROLE IDENTIFIERS
# ==========================================
MODEL_ROLE_A = "model_a"  # Dùng cho 5 agent (explorer / judge / scorer)
MODEL_ROLE_B = "model_b"  # Dùng cho Target Model (llm_target)


def _load_model_cfg(
    prefix: str,
    default_alias: str,
    default_baseline_url: str,
    default_turbo_url: str,
    default_gguf: str,
) -> dict:
    """
    Load cấu hình của 1 model từ biến môi trường (theo prefix).

    Mỗi model có 4 trường:
    - alias:        nhãn gửi trong field `model` của request (phải khớp --alias server)
    - gguf_file:    tên file .gguf nằm trong Factaudit/models/
    - baseline_url: endpoint server baseline (cache f32)
    - turbo_url:    endpoint server turboquant (cache turbo3/turbo4)

    Args:
        prefix: "MODEL_A" hoặc "MODEL_B"
        default_*: giá trị mặc định khi env chưa set

    Returns:
        dict cấu hình model
    """
    return {
        "alias": os.getenv(f"{prefix}_ALIAS", default_alias),
        "gguf_file": os.getenv(f"{prefix}_GGUF_FILE", default_gguf),
        "baseline_url": os.getenv(f"{prefix}_BASELINE_API_BASE", default_baseline_url),
        "turbo_url": os.getenv(f"{prefix}_TURBOQUANT_API_BASE", default_turbo_url),
    }


# ==========================================
# LLM FACTORY CLASS
# ==========================================
class LLMFactory:
    """
    Factory Pattern để tạo LLM instances với mode switching + 2 model độc lập.

    Hỗ trợ 2 chế độ mode (TOÀN CỤC, áp dụng cho cả 2 model):
    - "baseline":   Trỏ tới Baseline Server   (cache f32)
    - "turboquant": Trỏ tới TurboQuant+ Server (cache turbo3/turbo4)

    Và 2 model config:
    - model_a: dùng cho create_explorer / create_judge / create_scorer (5 agent)
    - model_b: dùng cho create_target (Target Model)

    Endpoint thực tế = chọn baseline_url/turbo_url của model tương ứng theo mode.

    Usage:
        factory = LLMFactory(mode="turboquant")
        llm_explorer = factory.create_explorer()   # -> Model A turboquant (8081)
        llm_target = factory.create_target()       # -> Model B turboquant (8083)
    """

    def __init__(self, mode: Optional[Literal["baseline", "turboquant", "auto"]] = None):
        """
        Khởi tạo LLMFactory.

        Args:
            mode: Chế độ LLM inference (TOÀN CỤC, áp dụng cho cả Model A & B)
                - "baseline":   Force dùng Baseline mode
                - "turboquant": Force dùng TurboQuant+ mode
                - "auto":       Tự động quyết định dựa trên USE_TURBOQUANT từ .env
                - None:         Giống như "auto"
        """
        self._mode = self._determine_mode(mode)
        self._api_key = os.getenv("API_KEY", "sk-not-required")
        self._timeout = int(os.getenv("TIMEOUT", "300"))

        # Load cấu hình 2 model độc lập từ env
        self._model_a = _load_model_cfg(
            "MODEL_A",
            default_alias="Qwen3-32B-Q8_0",
            default_baseline_url="http://127.0.0.1:8080/v1",
            default_turbo_url="http://127.0.0.1:8081/v1",
            default_gguf="Qwen3-32B-Q8_0.gguf",
        )
        self._model_b = _load_model_cfg(
            "MODEL_B",
            default_alias="Qwen3-14B-Q8_0",
            default_baseline_url="http://127.0.0.1:8082/v1",
            default_turbo_url="http://127.0.0.1:8083/v1",
            default_gguf="Qwen3-14B-Q8_0.gguf",
        )

        # Print mode info
        self._print_mode_info()

    def _determine_mode(self, mode: Optional[str]) -> str:
        """
        Xác định mode thực tế sẽ sử dụng.

        Priority (highest to lowest):
        1. Explicit mode parameter ("baseline" hoặc "turboquant")
        2. MODE environment variable
        3. USE_TURBOQUANT flag (nếu MODE=auto)
        """
        # 1. Explicit parameter has highest priority
        if mode in ["baseline", "turboquant"]:
            return mode

        # 2. Check MODE environment variable
        env_mode = os.getenv("MODE", "auto").lower()
        if env_mode in ["baseline", "turboquant"]:
            return env_mode

        # 3. Auto mode: check USE_TURBOQUANT flag
        use_turbo = os.getenv("USE_TURBOQUANT", "false").lower() == "true"
        return "turboquant" if use_turbo else "baseline"

    def _endpoint_for(self, cfg: dict) -> str:
        """
        Lấy API endpoint của 1 model dựa trên mode hiện tại (toàn cục).

        - turboquant -> cfg["turbo_url"]
        - baseline   -> cfg["baseline_url"]
        """
        if self._mode == "turboquant":
            return cfg["turbo_url"]
        return cfg["baseline_url"]

    @staticmethod
    def _truncate(text: str, width: int = 50) -> str:
        """Cắt chuỗi dài cho vừa ô hiển thị trong terminal."""
        text = str(text)
        if len(text) <= width:
            return text
        return "..." + text[-(width - 3):]

    def _gguf_status(self, gguf_file: str) -> tuple:
        """
        Tính đường dẫn + trạng thái file GGUF của 1 model (cho log/hiển thị).

        Returns:
            (path: Path, status_str: str)
        """
        gguf_path = resolve_gguf_model_path(gguf_file)
        if gguf_path.exists():
            status = f"found ({gguf_path.stat().st_size / (1024 ** 3):.1f} GB)"
        else:
            status = "NOT FOUND - server có thể không load được"
        return gguf_path, status

    def _print_mode_info(self):
        """In thông tin mode + cấu hình 2 model ra terminal để user tracking."""
        mode_display = "TurboQuant+ (KV Cache Compression)" if self._mode == "turboquant" else "Baseline (f32 cache)"

        path_a, status_a = self._gguf_status(self._model_a["gguf_file"])
        path_b, status_b = self._gguf_status(self._model_b["gguf_file"])
        base_a = self._endpoint_for(self._model_a)
        base_b = self._endpoint_for(self._model_b)

        print(f"┌" + "─" * 70 + "┐")
        print(f"│ {'LLM FACTORY INITIALIZED (2 models)':^66} │")
        print(f"├" + "─" * 70 + "┤")
        print(f"│ {'Mode:':<12}{self._truncate(mode_display):<56} │")
        print(f"├" + "─" * 70 + "┤")
        print(f"│ {'MODEL A (5 agent):':<70} │")
        print(f"│ {'Alias:':<12}{self._truncate(self._model_a['alias']):<56} │")
        print(f"│ {'API Base:':<12}{self._truncate(base_a):<56} │")
        print(f"│ {'GGUF File:':<12}{self._truncate(path_a.name):<56} │")
        print(f"│ {'GGUF Path:':<12}{self._truncate(path_a, 56):<56} │")
        print(f"│ {'GGUF Status:':<12}{self._truncate(status_a):<56} │")
        print(f"├" + "─" * 70 + "┤")
        print(f"│ {'MODEL B (Target):':<70} │")
        print(f"│ {'Alias:':<12}{self._truncate(self._model_b['alias']):<56} │")
        print(f"│ {'API Base:':<12}{self._truncate(base_b):<56} │")
        print(f"│ {'GGUF File:':<12}{self._truncate(path_b.name):<56} │")
        print(f"│ {'GGUF Path:':<12}{self._truncate(path_b, 56):<56} │")
        print(f"│ {'GGUF Status:':<12}{self._truncate(status_b):<56} │")

        # Print context size based on mode
        if self._mode == "turboquant":
            ctx_size = os.getenv("TURBOQUANT_CONTEXT_SIZE", "32768")
            print(f"│ {'Max Context:':<12}{self._truncate(ctx_size + ' tokens (4x capacity)'):<56} │")
        else:
            ctx_size = os.getenv("MAX_CONTEXT_SIZE", "8192")
            print(f"│ {'Max Context:':<12}{self._truncate(ctx_size + ' tokens'):<56} │")

        print(f"└" + "─" * 70 + "┘")

    @property
    def mode(self) -> str:
        """Get current mode (toàn cục, áp dụng cho cả Model A & B)."""
        return self._mode

    @property
    def api_base(self) -> str:
        """Get API endpoint của Model A (5 agent) theo mode hiện tại."""
        return self._endpoint_for(self._model_a)

    @property
    def api_base_target(self) -> str:
        """Get API endpoint của Model B (Target) theo mode hiện tại."""
        return self._endpoint_for(self._model_b)

    @property
    def alias_a(self) -> str:
        """Get alias của Model A (5 agent)."""
        return self._model_a["alias"]

    @property
    def alias_b(self) -> str:
        """Get alias của Model B (Target)."""
        return self._model_b["alias"]

    @property
    def gguf_path_a(self) -> Path:
        """Get resolved GGUF path của Model A (for logging/display)."""
        return resolve_gguf_model_path(self._model_a["gguf_file"])

    @property
    def gguf_path_b(self) -> Path:
        """Get resolved GGUF path của Model B (for logging/display)."""
        return resolve_gguf_model_path(self._model_b["gguf_file"])

    @property
    def gguf_path(self) -> Path:
        """Deprecated: alias của gguf_path_a (giữ cho tương thích ngược)."""
        return self.gguf_path_a

    def switch_mode(self, new_mode: Literal["baseline", "turboquant"]) -> None:
        """
        Switch runtime mode (toàn cục) cho cả Model A & B.

        Endpoint của từng model được derive on-demand từ mode, nên chỉ cần
        cập nhật self._mode (không cần cache _api_base).

        Args:
            new_mode: "baseline" hoặc "turboquant"
        """
        old_mode = self._mode
        self._mode = new_mode

        print(f"\n🔄 [LLMFactory] Mode switched: {old_mode.upper()} → {new_mode.upper()}")
        print(f"   Model A (agents): {self._endpoint_for(self._model_a)}")
        print(f"   Model B (target): {self._endpoint_for(self._model_b)}\n")

    # ==========================================
    # LLM CREATION METHODS
    # ==========================================

    def _create_base_llm(
        self,
        cfg: dict,
        temperature: float,
        max_tokens: Optional[int] = None,
        format: Optional[str] = None
    ) -> ChatOpenAI:
        """
        Tạo base ChatOpenAI instance cho 1 model cụ thể (cfg) + mode hiện tại.

        Instance trỏ tới endpoint của model đó (baseline_url hoặc turbo_url)
        tuỳ theo mode toàn cục của factory.

        Args:
            cfg: dict cấu hình model (_model_a hoặc _model_b)
            temperature: Temperature cho generation
            max_tokens: Maximum tokens (optional)
            format: Output format ("json" hoặc None)
                - Được GIỮ LẠI để tương thích ngược với signature của các agent.
                - Việc ép JSON output thực tế được xử lý qua .with_structured_output()
                  trong code agent; KHÔNG truyền response_format vào đây để tránh
                  xung đột với structured output (tool calling) của OpenAI-compatible API.

        Returns:
            ChatOpenAI instance
        """
        # Parse max_tokens
        if max_tokens is None:
            max_tokens = int(os.getenv("MAX_TOKENS", "4096"))

        # Build kwargs
        kwargs = {
            "model": cfg["alias"],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "api_key": self._api_key,
            "timeout": self._timeout,
        }

        # NOTE: giữ tham số `format` để không phá vỡ contract `create_judge(format="json")`.
        # JSON mode được xử lý qua with_structured_output() ở tầng agent, không qua đây.
        # (Truyền response_format={"type":"json_object"} có thể gây xung đột với tool
        #  calling mà with_structured_output() sử dụng -> cố tình bỏ qua.)
        _ = format

        return ChatOpenAI(base_url=self._endpoint_for(cfg), **kwargs)

    def _create_fallback_gemini(
        self,
        temperature: float,
        model: str = "gemini-2.5-flash"
    ) -> ChatGoogleGenerativeAI:
        """
        Tạo fallback Gemini instance (CLOUD, tuỳ chọn).

        CHỈ dùng khi có GEMINI_API_KEY và không muốn dùng server local.
        Không liên quan tới luồng GGUF/llama.cpp. Bỏ qua method này nếu
        không có key.

        Args:
            temperature: Temperature cho generation
            model: Gemini model name

        Returns:
            ChatGoogleGenerativeAI instance

        Raises:
            ValueError: nếu thiếu GEMINI_API_KEY
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY must be set in .env for Gemini fallback")

        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            api_key=api_key
        )

    # ------------------------------------------
    # EXPLORER LLM (Appraiser, Prober, Evaluator Phase 1) -> MODEL A
    # ------------------------------------------
    def create_explorer(self) -> ChatOpenAI:
        """
        Tạo Explorer LLM (Model A).

        Usage: Appraiser, Prober, Evaluator Phase 1
        Yêu cầu: Nhiệt độ cao (1.0) để tạo kịch bản mới lạ, sáng tạo
        """
        temp = float(os.getenv("TEMPERATURE_EXPLORER", "1.0"))

        llm = self._create_base_llm(self._model_a, temperature=temp)

        print(f"  ✓ llm_explorer created (Model A, temp={temp})")

        return llm

    # ------------------------------------------
    # JUDGE LLM (Inquirer, Quality Inspector) -> MODEL A
    # ------------------------------------------
    def create_judge(self, format: Optional[str] = "json") -> ChatOpenAI:
        """
        Tạo Judge LLM (Model A).

        Usage: Inquirer, Quality Inspector, Internal Judge
        Yêu cầu: Nhiệt độ thấp (0.0) để đảm bảo tính chính xác, công bằng

        Args:
            format: "json" (mặc định) - giữ lại cho tương thích; JSON mode
                thực sự xử lý qua with_structured_output() ở tầng agent.
        """
        temp = float(os.getenv("TEMPERATURE_JUDGE", "0.0"))

        llm = self._create_base_llm(self._model_a, temperature=temp, format=format)

        print(f"  ✓ llm_judge created (Model A, temp={temp}, format={format})")

        return llm

    # ------------------------------------------
    # SCORER LLM (Evaluator Phase 2) -> MODEL A
    # ------------------------------------------
    def create_scorer(self) -> ChatOpenAI:
        """
        Tạo Scorer LLM (Model A).

        Usage: Evaluator Phase 2
        Yêu cầu: Nhiệt độ thấp (0.0) để đánh giá nhất quán
        """
        temp = float(os.getenv("TEMPERATURE_SCORER", "0.0"))

        llm = self._create_base_llm(self._model_a, temperature=temp)

        print(f"  ✓ llm_scorer created (Model A, temp={temp})")

        return llm

    # ------------------------------------------
    # TARGET LLM (Model Under Test) -> MODEL B
    # ------------------------------------------
    def create_target(self) -> ChatOpenAI:
        """
        Tạo Target LLM (Model B - hoàn toàn độc lập với các agent).

        Usage: Mô hình bị kiểm toán
        Yêu cầu: Nhiệt độ thấp (0.0) để simulate user behavior
        """
        temp = float(os.getenv("TEMPERATURE_TARGET", "0.0"))

        llm = self._create_base_llm(self._model_b, temperature=temp)

        print(f"  ✓ llm_target created (Model B, temp={temp})")

        return llm

    # ------------------------------------------
    # BATCH CREATION (Create all LLMs at once)
    # ------------------------------------------
    def create_all(self) -> dict:
        """
        Tạo tất cả LLM instances và return dưới dạng dict.

        explorer/judge/scorer -> Model A; target -> Model B.

        Returns:
            dict với keys: 'explorer', 'judge', 'scorer', 'target'
        """
        print(f"\n[LLMFactory] Creating all LLM instances...")

        return {
            "explorer": self.create_explorer(),
            "judge": self.create_judge(),
            "scorer": self.create_scorer(),
            "target": self.create_target()
        }


# ==========================================
# GLOBAL LLM INSTANCES (Lazy Initialization)
# ==========================================
"""
Sử dụng pattern: Lazy Initialization + Singleton

Các LLM instances sẽ được tạo khi first access,
dựa trên mode được xác định tại thời điểm đó.

Để switch mode runtime:
1. Gọi switch_llm_mode(new_mode) (mode toàn cục, flip cả Model A & B)
"""

# Global factory instance (lazy loaded)
_llm_factory: Optional[LLMFactory] = None

# Global LLM instances (lazy loaded)
llm_explorer = None
llm_judge = None
llm_scorer = None
llm_target = None


def get_factory(mode: Optional[str] = None) -> LLMFactory:
    """
    Get singleton LLMFactory instance.

    Args:
        mode: Mode override (optional)

    Returns:
        LLMFactory instance
    """
    global _llm_factory

    if _llm_factory is None or (mode is not None and _llm_factory.mode != mode):
        _llm_factory = LLMFactory(mode=mode)

    return _llm_factory


def initialize_llms(mode: Optional[str] = None) -> dict:
    """
    Initialize tất cả LLM instances với mode specified (toàn cục).

    Args:
        mode: "baseline", "turboquant", hoặc None (use .env config)

    Returns:
        dict với các LLM instances (explorer/judge/scorer -> Model A,
        target -> Model B)
    """
    global llm_explorer, llm_judge, llm_scorer, llm_target

    factory = get_factory(mode=mode)
    llms = factory.create_all()

    # Update global instances
    llm_explorer = llms["explorer"]
    llm_judge = llms["judge"]
    llm_scorer = llms["scorer"]
    llm_target = llms["target"]

    return llms


def switch_llm_mode(new_mode: Literal["baseline", "turboquant"]) -> None:
    """
    Switch runtime mode (toàn cục) và reinitialize tất cả LLM instances.
    Mode flip áp dụng cho cả Model A (agent) và Model B (target).

    Args:
        new_mode: "baseline" hoặc "turboquant"
    """
    global _llm_factory

    print(f"\n{'='*70}")
    print(f"🔄 SWITCHING LLM MODE TO: {new_mode.upper()}  (Model A + Model B)")
    print(f"{'='*70}\n")

    # Reset factory
    _llm_factory = None

    # Reinitialize LLMs
    initialize_llms(mode=new_mode)

    print(f"\n✅ All LLM instances switched to {new_mode.upper()} mode!\n")


# ==========================================
# GLOBAL CONSTANTS (From Original Config)
# ==========================================

# Retry & Threshold Settings
MAX_RETRIES = 3              # Số lần lặp tối đa khi LLM bị Judge/Quality Inspector từ chối
MAX_WEB_CHECKS = 2           # Số lần tối đa được phép quay lại web_check_node để sửa lỗi trước khi bỏ qua
LOW_SCORE_THRESHOLD = 3.0    # Ngưỡng điểm để xếp một test case vào loại "Bad Case" (dành cho Evaluator)
MAX_ITERATIONS = 3           # Số vòng lặp tối đa của Prober (Iterative Probing) theo bài báo

# Context Size Settings (dynamically based on mode — chung cho cả 2 model)
def get_max_context_size() -> int:
    """Lấy max context size dựa trên current mode (chung cho cả Model A & B)."""
    factory = get_factory()
    if factory.mode == "turboquant":
        return int(os.getenv("TURBOQUANT_CONTEXT_SIZE", "32768"))
    else:
        return int(os.getenv("MAX_CONTEXT_SIZE", "8192"))


# ==========================================
# INITIALIZATION (EXPLICIT — KHÔNG auto-init khi import)
# ==========================================
# Việc khởi tạo LLM phải tường minh qua:
#   - main.py     : initialize_llms(mode=...)  (chạy TRƯỚC khi stream graph)
#   - runtime     : switch_llm_mode(new_mode)   (flip cả Model A & B)
# Các agent đọc LLM tại call-time qua `config.llm_*` (đã sửa trong src/*/),
# nên luôn thấy instance đúng với mode hiện hành, bất kể thứ tự import.
#
# LƯU Ý: sau khi bỏ auto-init, các global llm_explorer/llm_judge/llm_scorer/
# llm_target là None cho tới khi có lời gọi initialize_llms() / switch_llm_mode().
