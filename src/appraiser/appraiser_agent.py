"""
Appraiser Agent Graph Definition

This module implements the Appraiser agent as a LangGraph StateGraph.
The Appraiser is responsible for analyzing the current taxonomy of fact-checking
tasks, identifying weaknesses based on model performance (bad cases), and
proposing/judging new test scenarios to adaptively update the taxonomy.
"""

from typing import Literal
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END

import config
from config import MAX_RETRIES
from .appraiser_state import AppraiserState, AnalysisOutput, JudgeOutput
from .appraiser_prompts import analysis_prompt, judge_new_task_prompt
from langsmith import traceable


# ==== GRAPH NODES ====
@traceable(name="Appraiser_Node_Execution", run_type="chain")
def analyze_node(state: AppraiserState):
    """
    Analyze the current taxonomy and propose a new test scenario if needed.

    This node uses a creative LLM to review the lowest-scoring test cases
    and determine if a new, challenging sub-task should be added to the taxonomy.
    """

    print(f"\n[Appraiser] Analyzing taxonomy for: {state.get('main_task', 'Unknown')}...")

    analyzer = config.llm_explorer.with_structured_output(AnalysisOutput)
    prompt = PromptTemplate.from_template(analysis_prompt)
    chain = prompt | analyzer

    response = chain.invoke({
        "taxonomy": str(state.get("taxonomy_scores", {})),
        "bad_cases": state.get("bad_cases_formatted", ""),
        "main_task": state.get("main_task", "")
    })

    if response.is_stop:
        print("[Appraiser] Taxonomy is comprehensive. Reporting Stop.")
        return {"is_terminated": True, "current_new_task": None}

    print(f"[Appraiser] Proposed new task: {response.task_name}")
    return {
        "current_new_task": response.task_name,
        "current_explanation": response.explanation,
        "is_terminated": False
    }

def judge_node(state: AppraiserState):
    """
    Judge the suitability of the newly proposed test scenario.

    This node acts as a strict evaluator, ensuring the proposed task is
    meaningful, non-redundant, and text-only.
    """

    print(f"[Judge] Evaluating proposed task: {state.get('current_new_task')}...")

    judger = config.llm_judge.with_structured_output(JudgeOutput)
    prompt = PromptTemplate.from_template(judge_new_task_prompt)
    chain = prompt | judger

    # Extract only the list of sub-task names so the Judge is not distracted by scores
    sub_tasks = list(state.get("taxonomy_scores", {}).keys())
    taxonomy_dict = {state.get("main_task", ""): sub_tasks}

    response = chain.invoke({
        "taxonomy": str(taxonomy_dict),
        "new_point": state.get("current_new_task"),
        "main_task": state.get("main_task")
    })

    current_retries = state.get("retry_count", 0)

    if response.is_suitable:
        print("[Judge] Valid! The new task has been accepted.")
        return {
            "final_new_task": state.get("current_new_task"),
            "is_terminated": False,
        }

    else:
        print(f"[Judge] Rejected! Reason: {response.reason}")
        new_retry_count = current_retries + 1

        # If rejected too many times (MAX_RETRIES), terminate the process
        if new_retry_count >= MAX_RETRIES:
            print(f"[Judge] Rejected {MAX_RETRIES} times. Terminating task creation.")
            return {"retry_count": new_retry_count, "is_terminated": True}

        return {"retry_count": new_retry_count}


# ==== GRAPH EDGES ====

def route_after_analyze(state: AppraiserState) -> Literal["judge_node", "__end__"]:
    """
    Decides whether to route to the judge node or end the graph based on the analyzer's output.
    """
    if state.get("is_terminated"):
        return "__end__"
    return "judge_node"

def route_after_judge(state: AppraiserState) -> Literal["analyze_node", "__end__"]:
    """
    Decides whether to route back to the analyzer to retry proposing a task,
    or end the graph if the task is accepted/terminated.
    """
    if state.get("final_new_task") or state.get("is_terminated"):
        return "__end__"
    return "analyze_node"


# ==== GRAPH BUILDER ====
appraiser_builder = StateGraph(AppraiserState)

appraiser_builder.add_node("analyze_node", analyze_node)
appraiser_builder.add_node("judge_node", judge_node)

appraiser_builder.add_edge(START, "analyze_node")
appraiser_builder.add_conditional_edges("analyze_node", route_after_analyze)
appraiser_builder.add_conditional_edges("judge_node", route_after_judge)

appraiser_graph = appraiser_builder.compile()
