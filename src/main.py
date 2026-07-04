"""
===============================================
FACT-AUDIT Main Entry Point
===============================================
Module này là điểm khởi đầu của hệ thống FACT-AUDIT.

Nhiệm vụ:
1. Parse command-line arguments (argparse)
2. Initialize LLM instances với mode được chọn
3. Setup DualLogger để capture stdout/stderr
4. Run Master Graph orchestration

Usage:
    # Chạy với Baseline mode (default)
    python src/main.py

    # Chạy với TurboQuant+ mode
    python src/main.py --mode turboquant

    # Chạy với Auto mode (sử dụng config từ .env)
    python src/main.py --mode auto

    # Chạy với custom max iterations
    python src/main.py --mode turboquant --max-iterations 5
"""

import os
import sys
import datetime
import argparse
from dotenv import load_dotenv

# Import LLM initialization function
from config import (
    initialize_llms,
    switch_llm_mode,
    get_max_context_size,
    MAX_RETRIES,
    MAX_WEB_CHECKS,
    LOW_SCORE_THRESHOLD,
    MAX_ITERATIONS
)

# Import Master Graph
from main_graph import master_graph


# ==========================================
# ARGUMENT PARSER
# ==========================================
def parse_arguments():
    """
    Parse command-line arguments cho FACT-AUDIT.

    Returns:
        argparse.Namespace với các arguments
    """
    parser = argparse.ArgumentParser(
        description="""
        ╔═══════════════════════════════════════════════════════════════╗
        ║           FACT-AUDIT: Automated Fact-Checking Auditor        ║
        ║                    Multi-Agent LangGraph System                ║
        ╚═══════════════════════════════════════════════════════════════╝
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
          python src/main.py                           # Auto mode from .env
          python src/main.py --mode baseline           # Force Baseline mode
          python src/main.py --mode turboquant         # Force TurboQuant+ mode
          python src/main.py --mode turboquant -i 5    # TurboQuant with 5 iterations
          python src/main.py --verbose                 # Enable verbose logging
        """
    )

    # ==========================================
    # MODE SELECTION
    # ==========================================
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        choices=["baseline", "turboquant", "auto"],
        default="auto",
        help="""
        LLM Inference Mode (default: auto)
        - baseline:   Use Baseline server (Q8_0/FP16, 8K context)
        - turboquant: Use TurboQuant+ server (turbo3/turbo4, 32K context)
        - auto:       Use .env config (USE_TURBOQUANT flag)
        """
    )

    # ==========================================
    # ITERATION SETTINGS
    # ==========================================
    parser.add_argument(
        "--max-iterations",
        "-i",
        type=int,
        help=f"""
        Override MAX_ITERATIONS for Prober agent
        Default: {MAX_ITERATIONS}
        Paper recommends: 30
        """
    )

    parser.add_argument(
        "--max-retries",
        "-r",
        type=int,
        help=f"""
        Override MAX_RETRIES for LLM rejection retry
        Default: {MAX_RETRIES}
        """
    )

    # ==========================================
    # THRESHOLD SETTINGS
    # ==========================================
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        help=f"""
        Override LOW_SCORE_THRESHOLD for 'Bad Case' classification
        Default: {LOW_SCORE_THRESHOLD}
        """
    )

    # ==========================================
    # LOGGING OPTIONS
    # ==========================================
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory for log files (default: logs)"
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging (print all node states)"
    )

    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable file logging (console only)"
    )

    # ==========================================
    # CONTEXT OPTIONS
    # ==========================================
    parser.add_argument(
        "--context-size",
        "-c",
        type=int,
        help="""
        Override max context size (tokens)
        Baseline: 8192, TurboQuant: 32768
        """
    )

    return parser.parse_args()


# ==========================================
# DUAL LOGGER CLASS
# ==========================================
class DualLogger:
    """
    Class này hoạt động như một bộ chia (Tee):
    Mỗi khi hệ thống gọi lệnh print(), nó sẽ in ra màn hình (terminal)
    đồng thời ghi trực tiếp xuống file log.

    Features:
    - Auto-create log directory
    - Timestamp-based log filenames
    - UTF-8 encoding support
    - Immediate flush to prevent data loss
    """

    def __init__(self, log_dir="logs", verbose=False):
        """
        Khởi tạo DualLogger.

        Args:
            log_dir: Directory cho log files
            verbose: Enable verbose mode
        """
        # Tự động tạo thư mục logs nếu chưa có
        os.makedirs(log_dir, exist_ok=True)

        # Tạo tên file log dán nhãn theo thời gian chạy
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        mode_tag = _get_mode_tag()  # Lấy mode tag từ config
        self.log_filepath = os.path.join(
            log_dir,
            f"fact_audit_{mode_tag}_{timestamp}.log"
        )

        self.terminal = sys.stdout
        self.log_file = open(self.log_filepath, "a", encoding="utf-8")
        self.verbose = verbose

        # In thông báo để biết log đang được ghi ở đâu (chỉ in ra terminal)
        header = f"📁 Log đang được ghi tại: {self.log_filepath}\n"
        self.terminal.write(header)

    def write(self, message):
        """Ghi message ra cả terminal và file log."""
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()  # Ép ghi ngay xuống ổ cứng

    def flush(self):
        """Flush cả terminal và file."""
        self.terminal.flush()
        self.log_file.flush()


def _get_mode_tag() -> str:
    """Lấy mode tag cho filename."""
    try:
        from config import get_factory
        factory = get_factory()
        return factory.mode[:4].upper()  # BASE hoặc TURB
    except:
        return "AUTO"


# ==========================================
# PRINT BANNER
# ==========================================
def print_banner(args):
    """
    In banner hệ thống với đầy đủ thông tin cấu hình.

    Args:
        args: Parsed arguments from argparse
    """
    mode_display = {
        "baseline": "BASELINE (Q8_0/FP16)",
        "turboquant": "TURBOQUANT+ (KV Cache Compression)",
        "auto": "AUTO (dựa trên .env)"
    }

    current_mode = args.mode
    if current_mode == "auto":
        from config import get_factory
        factory = get_factory()
        actual_mode = factory.mode
        mode_label = f"AUTO → {actual_mode.upper()}"
    else:
        mode_label = current_mode.upper()

    separator = "┌" + "─" * 76 + "┐"

    print(f"\n{separator}")
    print(f"│ {'╔═════════════════════════════════════════════════════════════════════╗':^74} │")
    print(f"│ {'║           FACT-AUDIT: Automated Fact-Checking Auditor          ║':^74} │")
    print(f"│ {'║                    Multi-Agent LangGraph System                   ║':^74} │")
    print(f"│ {'╚═════════════════════════════════════════════════════════════════════╝':^74} │")
    print(f"{separator}")
    print(f"│ {'CONFIGURATION':^74} │")
    print(f"{separator}")
    print(f"│ {'Mode:':<20} {mode_display.get(current_mode, current_mode):<52} │")
    print(f"│ {'Log Directory:':<20} {args.log_dir:<52} │")
    print(f"│ {'Verbose:':<20} {'Enabled' if args.verbose else 'Disabled':<52} │")
    print(f"{separator}")

    # Print LLM info
    try:
        from config import get_factory
        factory = get_factory()

        if current_mode == "auto":
            print(f"│ {'Actual Mode:':<20} {factory.mode.upper() + ' mode':<52} │")

        # Hệ thống 2 model độc lập: Model A (5 agent) + Model B (Target).
        # Mode-switching là toàn cục -> cùng một mode áp dụng cho cả hai.
        print(f"│ {'Model A (agents):':<20} {factory.alias_a + ' @ ' + factory.api_base:<52} │")
        print(f"│ {'Model B (target):':<20} {factory.alias_b + ' @ ' + factory.api_base_target:<52} │")
        print(f"│ {'Max Context:':<20} {f'{get_max_context_size()} tokens':<52} │")
        print(f"{separator}")
    except Exception as e:
        print(f"│ {'LLM Config:':<20} {'Error loading config':<52} │")
        print(f"{separator}")

    # Print overridden settings
    overrides = []
    if args.max_iterations:
        overrides.append(f"MAX_ITERATIONS={args.max_iterations}")
    if args.max_retries:
        overrides.append(f"MAX_RETRIES={args.max_retries}")
    if args.threshold:
        overrides.append(f"LOW_SCORE_THRESHOLD={args.threshold}")
    if args.context_size:
        overrides.append(f"CONTEXT_SIZE={args.context_size}")

    if overrides:
        print(f"│ {'OVERRIDES:':^74} │")
        for override in overrides:
            print(f"│  • {override:<70} │")
        print(f"{separator}")

    print(f"│ {'':^74} │")
    print(f"└" + "─" * 76 + "┘")
    print()


# ==========================================
# MAIN RUN FUNCTION
# ==========================================
def run_fact_audit(args):
    """
    Main function chạy FACT-AUDIT system.

    Args:
        args: Parsed arguments from argparse
    """
    # ==========================================
    # BUILD INITIAL STATE
    # ==========================================
    initial_state = {
        "main_task": "Fact-Checking",
        "final_new_task": "Health and Medical Fake News",
        "categories": {
            "Fact-Checking": [
                "Health and Medical Fake News",
                "Political Rumors",
                "Financial Misinformation"
            ]
        },
        "taxonomy_scores": {
            "Health and Medical Fake News": 0.0,
        },
        "bad_cases_formatted": "",
        "is_terminated": False,
        "seed_data": [],
        "memory_pool": []
    }

    # ==========================================
    # RUN MASTER GRAPH
    # ==========================================
    print("🚀 BẮT ĐẦU KÍCH HOẠT HỆ THỐNG FACT-AUDIT 🚀")
    print("─" * 78)

    config = {"recursion_limit": 1000}

    try:
        for event in master_graph.stream(initial_state, config=config):
            for node_name, node_state in event.items():
                # In thông tin node completion (hoặc verbose mode)
                if args.verbose or node_name not in ["__start__", "__end__"]:
                    print(f"\n✅ [Node Hoàn Thành]: {node_name.upper()}")

                # Detail logging cho từng node
                if node_name == "inquirer_node":
                    seed_count = len(node_state.get('seed_data', []))
                    print(f"   -> Đã sinh {seed_count} seed cases. Đang đẩy vào Evaluation Subgraph...")

                elif node_name == "evaluation_subgraph":
                    print(f"   -> Đã hoàn tất 1 luồng 'tra tấn' Target LLM.")

                elif node_name == "aggregate_bad_cases_node":
                    bad_count = len(node_state.get('memory_pool', []))
                    print(f"   -> Đã gom {bad_count} cases. Chuyển cho Appraiser phân tích.")

                elif node_name == "appraiser_subgraph":
                    if node_state.get("is_terminated"):
                        print(f"   -> 🎯 Appraiser báo cáo Taxonomy đã hoàn hảo. DỪNG HỆ THỐNG!")
                    else:
                        new_task = node_state.get('final_new_task', 'Unknown')
                        print(f"   -> 🔄 Đã chốt kịch bản mới: {new_task}")
                        print(f"   -> Bắt đầu vòng lặp tiến hóa tiếp theo...")

        from compute_metrics import calculate_fact_audit_metrics
        calculate_fact_audit_metrics()
    except Exception as e:
        print(f"\n❌ LỖI TRONG QUÁ TRÌNH CHẠY: {e}")
        import traceback
        traceback.print_exc()


# ==========================================
# MAIN ENTRY POINT
# ==========================================
def main():
    """
    Main entry point cho FACT-AUDIT system.

    Flow:
    1. Parse arguments
    2. Load environment variables
    3. Initialize LLMs với mode được chọn
    4. Setup DualLogger
    5. Print banner
    6. Run fact audit
    """
    # ==========================================
    # STEP 1: PARSE ARGUMENTS
    # ==========================================
    args = parse_arguments()

    # ==========================================
    # STEP 2: LOAD ENVIRONMENT
    # ==========================================
    load_dotenv()

    if os.environ.get("LANGCHAIN_TRACING_V2") == "true":
        print("LangSmith Tracing is ENABLED.")
    # Validate required environment variables
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("⚠️  Warning: GEMINI_API_KEY not set. Using llama-cpp-turboquant instead.")
        print("   Set GEMINI_API_KEY in .env if you want to use Gemini as fallback.\n")

    # ==========================================
    # STEP 3: INITIALIZE LLMS WITH SELECTED MODE
    # ==========================================
    print("\n🔧 Initializing LLM instances...")
    initialize_llms(mode=args.mode if args.mode != "auto" else None)
    print("✅ LLM initialization complete!\n")

    # ==========================================
    # STEP 4: SETUP DUAL LOGGER
    # ==========================================
    if not args.no_log:
        logger = DualLogger(log_dir=args.log_dir, verbose=args.verbose)
        sys.stdout = logger
        sys.stderr = logger

    # ==========================================
    # STEP 5: PRINT BANNER
    # ==========================================
    print_banner(args)

    # ==========================================
    # STEP 6: RUN FACT AUDIT
    # ==========================================
    run_fact_audit(args)


# ==========================================
# SCRIPT ENTRY POINT
# ==========================================
if __name__ == "__main__":
    main()
