"""
Advanced Search Tools for Quality Inspector Agent

This module replaces the basic Wikipedia keyword matching with a robust
RAG-based web search approach using Tavily. It fetches real-time ground truth
from the web and uses a deterministic LLM to verify if the AI-generated
evidence is factual or hallucinated.
"""

import json
from typing import Tuple
from pydantic import BaseModel, Field
# from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_tavily import TavilySearch
from langchain_core.prompts import PromptTemplate
import config
from .inspector_prompts import verification_prompt
from langsmith import traceable

tavily_search = TavilySearch(max_results=3, search_depth="advanced")

class WebVerification(BaseModel):
    """Schema for evaluating generated evidence against real web search results."""
    is_factual: bool = Field(
        description="True if the generated evidence matches the real web search results. False if it is fabricated, hallucinated, or contradicts reality."
    )
    correction: str = Field(
        default="",
        description="If is_factual is False, provide the actual ground truth facts extracted from the web results. Leave empty if True."
    )

@traceable(name="Advanced_Tavily_Web_Check", run_type="tool")
def advanced_web_check(test_case: dict) -> Tuple[bool, str]:
    """
    Main function to validate a test case using Tavily Search and LLM Verification.
    Only applies if test_mode is [evidence].

    Returns:
        Tuple[bool, str]: (is_passed, correction_feedback)
    """
    test_mode = test_case.get("test_mode", "")
    if "[evidence]" not in test_mode:
        # Skip web check for [claim] or [wisdom of crowds] modes
        return True, ""

    prompt_content = test_case.get("prompt", {})
    claim = prompt_content.get("source_claim", "")
    evidence = prompt_content.get("auxiliary_info", "")

    if not evidence or not isinstance(evidence, str):
        return True, ""

    try:
        # Query the web using a combination of the claim and a snippet of the evidence
        search_query = f"{claim} {evidence[:100]}"
        print(f"[WebTool] Searching Tavily for: '{search_query}'...")
        docs = tavily_search.invoke({"query": search_query})

        # Lớp xử lý an toàn: Ép kiểu chuỗi JSON về dạng danh sách (List[Dict])
        if isinstance(docs, str):
            try:
                docs = json.loads(docs)
            except json.JSONDecodeError:
                pass # Giữ nguyên nếu không thể phân tích chuỗi

        # Phân rã dữ liệu an toàn dựa trên loại dữ liệu cuối cùng
        if isinstance(docs, list):
            web_results_str = "\n\n".join([f"Source: {doc.get('url', 'N/A')}\nContent: {doc.get('content', '')}" for doc in docs if isinstance(doc, dict)])
        else:
            web_results_str = str(docs) # Dự phòng nếu dữ liệu vẫn là chuỗi thuần

        # Use the Judge LLM to cross-reference evidence with web facts
        verifier = config.llm_judge.with_structured_output(WebVerification)
        prompt = PromptTemplate.from_template(verification_prompt)

        print("[WebTool] Evaluating evidence against search results...")
        result: WebVerification = (prompt | verifier).invoke({
            "claim": claim,
            "evidence": evidence,
            "web_results": web_results_str
        })

        return result.is_factual, result.correction

    except Exception as e:
        print(f"[WebTool] Error during web search or LLM verification: {e}")
        return False, f"Verification system error: {str(e)}"