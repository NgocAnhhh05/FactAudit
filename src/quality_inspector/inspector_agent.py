import json
from typing import Literal
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END

import config
from config import MAX_RETRIES
from .inspector_state import InspectorState, InspectionOutput
from .inspector_prompts import judge_new_case_prompt
from .tools import advanced_web_check
from langsmith import traceable

# ==== GRAPH NODES ====

# def select_next_case_node(state: InspectorState):
#     """Lấy test case tiếp theo từ danh sách pending_cases để xử lý."""
#     pending = state.get("pending_cases", [])
#     approved = state.get("approved_cases", [])

#     if not pending:
#         # Hết dữ liệu để duyệt
#         return {"current_case": None}

#     current = pending[0]
#     remaining_pending = pending[1:]

#     print(f"\n[Inspector] Đang lấy 1 test case ra để kiểm tra. Còn lại {len(remaining_pending)} cases.")
#     return {
#         "current_case": current,
#         "pending_cases": remaining_pending,
#         "retry_count": 0  # Reset retry count cho case mới
#     }

@traceable(name="Inspector_Web_Check_Node", run_type="chain")
def web_check_node(state: InspectorState):
    """Tầng 1: Kiểm tra thực tế bằng Tavily Search và Mini-RAG."""
    current_case = state["current_case"]
    print(f"\n[Inspector] Đang kiểm tra thực tế trên Web cho mode {current_case.get('test_mode')}...")

    is_factual, correction_info = advanced_web_check(current_case)

    if not is_factual:
        print("[Inspector] Phát hiện ảo giác so với thực tế mạng!")
        # Trực tiếp bơm sự thật (ground truth) cào được từ web vào lỗi
        # Để LLM ở node tiếp theo lấy thông tin này viết lại evidence cho chuẩn
        current_case["web_error"] = f"Web Fact-Check Failed! The generated evidence is hallucinated. Use these REAL facts to rewrite the evidence: {correction_info}"
        return {"current_case": current_case}

    print("[Inspector] Bằng chứng khớp với thực tế mạng (Pass).")
    return {"current_case": current_case}

@traceable(name="Inspector_LLM_Review_Node", run_type="chain")
def llm_inspection_node(state: InspectorState):
    """Tầng 2: LLM thẩm định tinh và sửa lỗi."""
    current_case = state["current_case"]
    task_name = state.get("task_name", "Unknown")

    # # === SỬA BUG 2: FORCE DỌN DẸP DATA BẰNG PYTHON ===
    # # Đảm bảo auxiliary_info phải rỗng đối với chế độ [claim]
    # if current_case.get("test_mode") == "[claim]":
    #     if "prompt" in current_case:
    #         current_case["prompt"]["auxiliary_info"] = ""
    # # ===================================================

    # === SỬA BUG 2: XÓA SỔ HOÀN TOÀN KEY AUXILIARY_INFO ===
    if current_case.get("test_mode") == "[claim]":
        if "prompt" in current_case and "auxiliary_info" in current_case["prompt"]:
            del current_case["prompt"]["auxiliary_info"]

    inspector = config.llm_judge.with_structured_output(InspectionOutput)
    prompt = PromptTemplate.from_template(judge_new_case_prompt)
    chain = prompt | inspector

    web_error_msg = current_case.pop("web_error", None)
    case_str = json.dumps(current_case, ensure_ascii=False, indent=2)
    if web_error_msg:
        case_str = f"ATTENTION ERROR TO FIX: {web_error_msg}\n\n{case_str}"

    print("[LLM Inspector] Đang thẩm định tính hợp lệ...")
    response: InspectionOutput = chain.invoke({
        "task_name": task_name,
        "test_case_json": case_str
    })

    if response.is_valid and not web_error_msg:
        print("[LLM Inspector] Case đạt chuẩn! Tiếp tục tới Target LLM & Evaluator.")
        # QUAN TRỌNG: KHÔNG set current_case = None.
        # Trả về case sạch sẽ (không còn cờ error) để Target LLM dùng
        return {"current_case": response.revised_case.model_dump(), "retry_count": 0}
    else:
        print(f"[LLM Inspector] Vi phạm quy tắc. Feedback: {response.feedback}")

        # Gắn cờ 'llm_error' vào bản revised để Router ở main_graph nhận diện được
        revised = response.revised_case.model_dump()
        revised["llm_error"] = response.feedback

        return {
            "current_case": revised,
            "retry_count": state.get("retry_count", 0) + 1
        }

# ==== GRAPH EDGES ====

@traceable(name="Inspector_Route_After_Select", run_type="chain")
def route_after_select(state: InspectorState) -> Literal["web_check_node", "__end__"]:
    """Nếu không còn current_case (hết mảng), kết thúc đồ thị."""
    if not state.get("current_case"):
        return "__end__"
    return "web_check_node"

@traceable(name="Inspector_Route_After_Inspection", run_type="chain")
def route_after_inspection(state: InspectorState) -> Literal["select_next_case_node", "web_check_node"]:
    """
    Quyết định:
    - Nếu current_case rỗng (nghĩa là đã approve xong), quay lại Select Node bốc case tiếp theo.
    - Nếu current_case bị LLM bắt lỗi và sửa lại, quay lại Web Check Node để kiểm tra lại case vừa sửa.
    """
    if not state.get("current_case"):
        return "select_next_case_node"

    if state.get("retry_count", 0) >= MAX_RETRIES:
        print(f"[Inspector] Đã sửa quá {MAX_RETRIES} lần nhưng vẫn lỗi. Bỏ qua test case này!")
        return "select_next_case_node"

    return "web_check_node"

# ==== GRAPH BUILDER ====
# inspector_builder = StateGraph(InspectorState)

# inspector_builder.add_node("select_next_case_node", select_next_case_node)
# inspector_builder.add_node("web_check_node", web_check_node)
# inspector_builder.add_node("llm_inspection_node", llm_inspection_node)

# inspector_builder.add_edge(START, "select_next_case_node")
# inspector_builder.add_conditional_edges("select_next_case_node", route_after_select)
# inspector_builder.add_edge("web_check_node", "llm_inspection_node")
# inspector_builder.add_conditional_edges("llm_inspection_node", route_after_inspection)

# inspector_graph = inspector_builder.compile()