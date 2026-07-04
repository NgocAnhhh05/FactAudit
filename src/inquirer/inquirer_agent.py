"""
Inquirer Agent Graph Definition

This module implements the Inquirer agent as a LangGraph StateGraph.
The Inquirer is responsible for generating prototype test data (seed data)
based on a specific test scenario, using a temperature=0.0 LLM to ensure
reproducibility and fairness.
"""

from typing import Literal
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END

from langsmith import traceable

import config
from config import MAX_RETRIES
from .inquirer_state import InquirerState, InquirerOutput
from .inquirer_prompts import gen_seed_prompt

# ==== GRAPH NODES ====

@traceable(name="Inquirer_Generate_Seed", run_type="chain")
def generate_seed_node(state: InquirerState):
    """
    Generate 10 prototype test cases based on the given task name.
    """
    task_name = state.get("final_new_task", "Unknown Task")
    print(f"\n[Inquirer] Generating seed data for scenario: {task_name} (Attempt {state.get('retry_count', 0) + 1})...")

    # Use the deterministic LLM (temp=0.0) with strictly structured output
    generator = config.llm_judge.with_structured_output(InquirerOutput)
    prompt = PromptTemplate.from_template(gen_seed_prompt)
    chain = prompt | generator

    try:
        response: InquirerOutput = chain.invoke({
            "categories": str(state.get("categories", {})),
            "task_name": task_name
        })

        # Convert Pydantic objects back to standard dictionaries for the global state
        seed_data_dicts = [test_case.model_dump() for test_case in response.test_cases]

        print(f"[Inquirer] Successfully generated {len(seed_data_dicts)} test cases.")
        return {"seed_data": seed_data_dicts}

    except Exception as e:
        print(f"[Inquirer] Generation failed or format was incorrect: {e}")
        # If the LLM fails to output the exact Pydantic schema, we catch it here
        return {"retry_count": state.get("retry_count", 0) + 1}


# ==== GRAPH EDGES ====

@traceable(name="Inquirer_Routing_Logic", run_type="chain")
def route_after_generation(state: InquirerState) -> Literal["generate_seed_node", "__end__"]:
    """
    Check if the seed data was generated successfully. If not, retry up to MAX_RETRIES.
    """
    if state.get("seed_data") and len(state["seed_data"]) > 0:
        return "__end__"

    if state.get("retry_count", 0) >= MAX_RETRIES:
        print(f"[Inquirer] Max retries ({MAX_RETRIES}) reached. Terminating generation.")
        return "__end__"

    return "generate_seed_node"


# ==== GRAPH BUILDER ====

inquirer_builder = StateGraph(InquirerState)

inquirer_builder.add_node("generate_seed_node", generate_seed_node)

inquirer_builder.add_edge(START, "generate_seed_node")
inquirer_builder.add_conditional_edges("generate_seed_node", route_after_generation)

inquirer_graph = inquirer_builder.compile()