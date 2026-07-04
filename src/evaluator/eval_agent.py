"""
Evaluator Agent Graph Definition (Phase 1 & Phase 2)

- Phase 1: Parallel generation -> Voting -> Refinement.
- Phase 2: 1-10 Scoring based on strict logical constraints.
"""

from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END

import config
from .eval_state import Phase1State, ReferenceOutput, VoteOutput, RefineOutput, ScoreOutput
from .eval_prompts import gen_fact_problem_prompt, gen_vote_prompt, judge_ref_answer_prompt, get_llm_score_prompt
from langsmith import traceable


@traceable(name="Eval_Extract_Prompt_Data", run_type="tool")
def _extract_prompt_data(current_case: dict) -> tuple:
    """Hàm tiện ích bóc tách dữ liệu từ Test Case"""
    prompt_data = current_case.get("prompt", {})
    return (
        prompt_data.get("source_claim", ""),
        prompt_data.get("auxiliary_info", ""),
        current_case.get("key_point", ""),
        current_case.get("test_mode", "")
    )


@traceable(name="Eval_Build_Question_Context", run_type="tool")
def _build_question_context(claim: str, aux_info: str) -> str:
    """Hàm gộp nội dung thành biến {question} cho prompt"""
    if aux_info and aux_info.strip():
        return f"Claim: {claim}\nContext: {aux_info}"
    return f"Claim: {claim}"


def gen_ref_1_node(state: Phase1State):
    claim, aux_info, _, _ = _extract_prompt_data(state["current_case"])
    question_context = _build_question_context(claim, aux_info)

    # CHÚ Ý: Đã đổi llm_explorer thành llm_judge (temp=0.0) theo đúng source code
    chain = PromptTemplate.from_template(gen_fact_problem_prompt) | config.llm_judge.with_structured_output(ReferenceOutput)
    res = chain.invoke({"question": question_context})
    return {"ref_ans_1": f"[{res.verdict}] {res.justification}"}

def gen_ref_2_node(state: Phase1State):
    claim, aux_info, _, _ = _extract_prompt_data(state["current_case"])
    question_context = _build_question_context(claim, aux_info)

    chain = PromptTemplate.from_template(gen_fact_problem_prompt) | config.llm_judge.with_structured_output(ReferenceOutput)
    res = chain.invoke({"question": question_context})
    return {"ref_ans_2": f"[{res.verdict}] {res.justification}"}

def gen_ref_3_node(state: Phase1State):
    claim, aux_info, _, _ = _extract_prompt_data(state["current_case"])
    question_context = _build_question_context(claim, aux_info)

    chain = PromptTemplate.from_template(gen_fact_problem_prompt) | config.llm_judge.with_structured_output(ReferenceOutput)
    res = chain.invoke({"question": question_context})
    return {"ref_ans_3": f"[{res.verdict}] {res.justification}"}


@traceable(name="Phase1_Voting_Process", run_type="chain")
def vote_node(state: Phase1State):
    print("\n[Phase 1] Đang biểu quyết (Voting) 3 luồng...")
    claim, aux_info, _, _ = _extract_prompt_data(state["current_case"])
    question_context = f"Claim: {claim}\nContext: {aux_info}"

    chain = PromptTemplate.from_template(gen_vote_prompt) | config.llm_judge.with_structured_output(VoteOutput)
    res = chain.invoke({
        "question": question_context,
        "ref_1": state["ref_ans_1"],
        "ref_2": state["ref_ans_2"],
        "ref_3": state["ref_ans_3"]
    })

    return {"voted_answer": f"[{res.verdict}] {res.justification}"}

@traceable(name="Phase1_Refinement", run_type="chain")
def refine_node(state: Phase1State):
    """Bước thanh lọc cuối cùng: Kiểm tra xem đáp án có bám sát key_point không"""
    print("[Phase 1] Đang thanh lọc (Refining) đáp án bám sát Key Point...")
    claim, aux_info, key_point, _ = _extract_prompt_data(state["current_case"])
    prompt_context = f"Claim: {claim}\nContext: {aux_info}"

    chain = PromptTemplate.from_template(judge_ref_answer_prompt) | config.llm_judge.with_structured_output(RefineOutput)
    res = chain.invoke({
        "answer": state["voted_answer"],
        "prompt": prompt_context,
        "key_point": key_point
    })

    print("[Phase 1] Đã chốt xong Gold Reference Answer!")
    return {"reference_answer": res.refined_answer}

# Xây dựng Phase 1 Sub-graph
phase1_builder = StateGraph(Phase1State)

phase1_builder.add_node("gen_ref_1", gen_ref_1_node)
phase1_builder.add_node("gen_ref_2", gen_ref_2_node)
phase1_builder.add_node("gen_ref_3", gen_ref_3_node)
phase1_builder.add_node("vote_node", vote_node)
phase1_builder.add_node("refine_node", refine_node)

# Điều hướng
phase1_builder.add_edge(START, "gen_ref_1")
phase1_builder.add_edge(START, "gen_ref_2")
phase1_builder.add_edge(START, "gen_ref_3")

# LangGraph tự động block tại vote_node cho đến khi 3 luồng trên xong
phase1_builder.add_edge(["gen_ref_1", "gen_ref_2", "gen_ref_3"], "vote_node")
phase1_builder.add_edge("vote_node", "refine_node")
phase1_builder.add_edge("refine_node", END)

evaluator_phase1_graph = phase1_builder.compile()


# ==========================================
# PHASE 2: SCORING NODE
# ==========================================

@traceable(name="Phase2_Score_Target_Response", run_type="chain")
def evaluator_phase2_score_node(state: dict):
    print("\n[Phase 2] Đang chấm điểm Target LLM (Thang 1-10)...")

    current_case = state.get("current_case", {})
    claim, aux_info, key_point, _ = _extract_prompt_data(current_case)
    question_context = f"Claim: {claim}\nContext: {aux_info}"

    target_response = state.get("target_response", "")
    reference_answer = state.get("reference_answer", "")

    chain = PromptTemplate.from_template(get_llm_score_prompt) | config.llm_scorer.with_structured_output(ScoreOutput)

    res: ScoreOutput = chain.invoke({
        "question": question_context,
        "key_point": key_point,
        "ref_answer": reference_answer,
        "target_response": target_response
    })

    print(f"[Phase 2] Chấm điểm xong! Score: {res.score}/10.0")

    return {
        "score": res.score,
        "comparison": res.comparison
    }