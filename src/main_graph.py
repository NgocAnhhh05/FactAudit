"""
Master Workflow for FACT-AUDIT

This module connects all agents into a fully automated, nested StateGraph.
It utilizes LangGraph's Send API for parallel map-reduce (Fan-out/Fan-in)
during the evaluation phase.
"""

import operator
from typing import Annotated, List, Dict, Any
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from evaluator.metrics_logger import log_case_to_json

# ==== IMPORT GLOBAL CONFIG ====
from config import MAX_ITERATIONS, MAX_WEB_CHECKS

from appraiser.appraiser_agent import appraiser_graph
from inquirer.inquirer_agent import generate_seed_node
from quality_inspector.inspector_agent import web_check_node, llm_inspection_node
from target_model.target_agent import target_llm_node
from evaluator.eval_agent import evaluator_phase1_graph, evaluator_phase2_score_node
from prober.prober_agent import prober_node


# ==========================================
# PHẦN 1: EVALUATION SUB-GRAPH (VÒNG LẶP TRA TẤN)
# ==========================================

class EvaluationState(TypedDict):
    task_name: str
    current_case: dict
    target_response: str
    reference_answer: str
    score: float
    comparison: str
    iteration_count: int
    memory_pool: Annotated[List[Dict], operator.add]
    retry_count: int

def save_memory_node(state: EvaluationState):
    record = {
        "prompt": state["current_case"].get("prompt", {}),
        "test_mode": state["current_case"].get("test_mode", ""),
        "key_point": state["current_case"].get("key_point", ""),
        "target_response": state.get("target_response", ""),
        "reference_answer": state.get("reference_answer", ""),
        "score": state.get("score", 0.0),
        "comparison": state.get("comparison", "")
    }
    log_case_to_json(record)
    return {"memory_pool": [record]}

def route_after_inspection(state: EvaluationState):
    current_case = state.get("current_case", {})
    if current_case.get("web_error") or current_case.get("llm_error"):
        if state.get("retry_count", 0) >= MAX_WEB_CHECKS:
            print("[Warning] Quá số lần retry ở Inspector. Cho qua với lỗi bảo lưu.")
            return ["target_llm_node", "evaluator_phase1_subgraph"]
        return "web_check_node"
    return ["target_llm_node", "evaluator_phase1_subgraph"]

def route_prober_loop(state: EvaluationState):
    if state["iteration_count"] < MAX_ITERATIONS:
        return "prober_node"
    return END

# --- Build Evaluation Sub-graph ---
eval_builder = StateGraph(EvaluationState)
eval_builder.add_node("web_check_node", web_check_node)
eval_builder.add_node("llm_inspection_node", llm_inspection_node)
eval_builder.add_node("target_llm_node", target_llm_node)
eval_builder.add_node("evaluator_phase1_subgraph", evaluator_phase1_graph)
eval_builder.add_node("evaluator_phase2_score_node", evaluator_phase2_score_node)
eval_builder.add_node("save_memory_node", save_memory_node)
eval_builder.add_node("prober_node", prober_node)

eval_builder.add_edge(START, "web_check_node")
eval_builder.add_edge("web_check_node", "llm_inspection_node")
eval_builder.add_conditional_edges(
    "llm_inspection_node",
    route_after_inspection,
    ["target_llm_node", "evaluator_phase1_subgraph", "web_check_node"]
)
eval_builder.add_edge(["target_llm_node", "evaluator_phase1_subgraph"], "evaluator_phase2_score_node")
eval_builder.add_edge("evaluator_phase2_score_node", "save_memory_node")
eval_builder.add_conditional_edges(
    "save_memory_node",
    route_prober_loop,
    {"prober_node": "prober_node", END: END}
)
eval_builder.add_edge("prober_node", "web_check_node")

evaluation_subgraph = eval_builder.compile()


# === SỬA BUG 3: WRAPPER NODE NGĂN RÒ RỈ STATE TRONG ĐA LUỒNG ===
def evaluation_wrapper_node(state: dict):
    """
    Node vỏ bọc: Gọi chạy đồ thị con và CHỈ lọc lại biến `memory_pool`
    để trả về Main Graph, tránh gây xung đột (concurrent update) các biến cục bộ.
    """
    # LangGraph cho phép chạy Sub-graph như một hàm qua .invoke()
    result = evaluation_subgraph.invoke(state)
    return {"memory_pool": result.get("memory_pool", [])}
# ===============================================================


# ==========================================
# PHẦN 2: MAIN GRAPH (VÒNG LẶP TIẾN HÓA)
# ==========================================

class MainState(TypedDict):
    main_task: str
    categories: Dict[str, List[str]]
    taxonomy_scores: Dict[str, float]
    bad_cases_formatted: str
    current_new_task: str
    final_new_task: str
    is_terminated: bool
    seed_data: List[Dict]
    memory_pool: Annotated[List[Dict], operator.add]

def aggregate_bad_cases_node(state: MainState):
    all_records = state.get("memory_pool", [])
    bad_cases = [r for r in all_records if r.get("score", 10.0) <= 3.0]

    formatted = f"Found {len(bad_cases)} bad cases in recent evaluation.\n"
    for bc in bad_cases[:10]:
        formatted += f"Prompt: {bc['prompt']}\nScore: {bc['score']}\nComment: {bc['comparison']}\n\n"

    return {"bad_cases_formatted": formatted}

def route_appraiser_to_inquirer(state: MainState):
    if state.get("is_terminated"):
        return END
    return "inquirer_node"

def route_fan_out_evaluations(state: MainState):
    task_name = state.get("final_new_task") or "Default_Task"
    seed_cases = state.get("seed_data", [])

    print(f"\n[Main] Đang Fan-out {len(seed_cases)} test cases vào Evaluation Workflow...")

    # SỬA BUG 3: Trỏ Send vào wrapper node thay vì trực tiếp vào subgraph
    return [Send("evaluation_wrapper", {
        "task_name": task_name,
        "current_case": case,
        "memory_pool": [],
        "iteration_count": 0
    }) for case in seed_cases]


# --- Build Main Graph ---
main_builder = StateGraph(MainState)

main_builder.add_node("appraiser_subgraph", appraiser_graph)
main_builder.add_node("inquirer_node", generate_seed_node)
main_builder.add_node("evaluation_wrapper", evaluation_wrapper_node) # Đăng ký node wrapper
main_builder.add_node("aggregate_bad_cases_node", aggregate_bad_cases_node)

main_builder.add_edge(START, "inquirer_node")

main_builder.add_conditional_edges(
    "appraiser_subgraph",
    route_appraiser_to_inquirer,
    {"inquirer_node": "inquirer_node", END: END}
)

main_builder.add_conditional_edges(
    "inquirer_node",
    route_fan_out_evaluations,
    ["evaluation_wrapper"] # Đổi đích thành wrapper
)

main_builder.add_edge("evaluation_wrapper", "aggregate_bad_cases_node") # Đổi luồng ra từ wrapper
main_builder.add_edge("aggregate_bad_cases_node", "appraiser_subgraph")

master_graph = main_builder.compile()