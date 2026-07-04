"""
Prober Agent Node

Reads the Memory Pool, performs importance sampling (mixing good and bad cases),
and generates a new, highly complex test case to probe the Target LLM's weaknesses.
"""

import random
from typing import Literal
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

# Nhớ dùng llm_explorer (temp=1.0) để Prober có sức sáng tạo đi săn mìn
import config
from .prober_prompt import deep_search_prompt

from langsmith import traceable

from typing import List, Dict, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field, model_validator

# ==== ĐỊNH NGHĨA SCHEMA CỐT LÕI MỚI ====
class PromptContent(BaseModel):
    source_claim: str = Field(description="The statement to be fact-checked.")
    auxiliary_info: str = Field(default="", description="External knowledge source or empty.")

class TestCase(BaseModel):
    key_point: str = Field(description="Short sentence summarizing the specific, detailed key point.")
    test_mode: Literal["[claim]", "[evidence]", "[wisdom of crowds]"] = Field(description="Problem setting for fact-checking.")
    prompt: PromptContent = Field(description="Core content.")

    @model_validator(mode="after")
    def enforce_fact_audit_criteria(self) -> "TestCase":
        mode = self.test_mode
        context = self.prompt.auxiliary_info or ""

        # Tiêu chí 3: Chế độ [claim] bắt buộc trống ngữ cảnh
        if mode == "[claim]":
            if context.strip() != "":
                self.prompt.auxiliary_info = "" # Tự động dọn dẹp

        # Tiêu chí 5: Chế độ [evidence] bắt buộc phải có bằng chứng
        elif mode == "[evidence]":
            if not context.strip():
                raise ValueError("For '[evidence]' mode, 'auxiliary_info' cannot be empty. Please provide some evidence.")

        return self

# ==== ĐẦU RA CỦA INSPECTOR ====
class InspectionOutput(BaseModel):
    is_valid: bool = Field(description="True if the test case is highly factual and meets all formatting criteria.")
    feedback: str = Field(description="Detailed feedback to fix the case if invalid, empty if valid.")
    revised_case: TestCase = Field(description="The corrected test case according to facts and formatting criteria.")

# ==== TRẠNG THÁI CỦA ĐỒ THỊ INSPECTOR ====
class InspectorState(TypedDict):
    task_name: str
    current_case: Dict
    retry_count: int


# ==== LOGIC SAMPLING TỪ SOURCE CODE ====
@traceable(name="Prober_Sample_History", run_type="tool")
def _sample_history(memory_pool: list, show_num: int = 5) -> list:
    """
    Thuật toán Importance Sampling: Ưu tiên bốc Bad Cases để LLM khoét sâu điểm yếu.
    """
    if not memory_pool:
        return []

    good_cases = [item for item in memory_pool if item.get('score', 10.0) > 3.0]
    bad_cases = [item for item in memory_pool if item.get('score', 10.0) <= 3.0]

    sample_his = []

    # Giống mã nguồn: Nếu ít data quá thì bốc 5 câu gần nhất
    if len(good_cases) < 5 or len(bad_cases) < 2:
        sample_his = memory_pool[-show_num:]
    else:
        # Bốc ngẫu nhiên 2 bad cases
        sample_his = random.sample(bad_cases, 2)
        retry = 50
        # Cố gắng nhồi thêm 3 good cases cho đủ 5
        while len(sample_his) < 5 and retry > 0:
            retry -= 1
            good_case = random.choice(good_cases)
            if good_case not in sample_his:
                sample_his.append(good_case)

        if len(sample_his) < 5:
            sample_his = memory_pool[-show_num:]

    # Sắp xếp theo điểm từ cao xuống thấp để nhét vào prompt
    return sorted(sample_his, key=lambda k: k.get('score', 0), reverse=True)


# ==== PROBER NODE ====
@traceable(name="Prober_Generate_Complex_Case", run_type="chain")
def prober_node(state: dict):
    print(f"\n[Prober] Vòng lặp thứ {state.get('iteration_count', 0) + 1}. Đang phân tích điểm yếu...")

    memory_pool = state.get("memory_pool", [])
    task_name = state.get("task_name", "Fact-Checking Task")

    # 1. Bốc mẫu lịch sử
    sampled_history = _sample_history(memory_pool)

    # 2. Ép chuỗi lịch sử thành Context
    history_str = ""
    for j in sampled_history:
        prompt_data = j.get('prompt', {})
        claim = prompt_data.get('source_claim', '')
        history_str += f"Prompt: {claim}\n"
        history_str += f"Test Mode: {j.get('test_mode', '')}\n"
        history_str += f"Key Point: {j.get('key_point', '')}\n"
        history_str += f"Target Answer: {j.get('target_response', '')}\n"
        history_str += f"Comment: {j.get('comparison', '')}\n"
        history_str += f"Score: {j.get('score', '')}\n\n"

    # 3. Gọi LLM sinh câu hỏi xoắn não hơn
    prober = config.llm_explorer.with_structured_output(TestCase)
    chain = PromptTemplate.from_template(deep_search_prompt) | prober

    try:
        res: TestCase = chain.invoke({
            "history_context": history_str,
            "task_name": task_name
        })

        new_case_dict = res.model_dump()
        print(f"[Prober] Đã đẻ xong câu hỏi mới! Key Point: {new_case_dict['key_point']}")

        # 4. Trả về Test Case mới để ghi đè 'current_case', sẵn sàng ném cho Quality Inspector
        # Tăng biến đếm vòng lặp
        return {
            "current_case": new_case_dict,
            "iteration_count": state.get("iteration_count", 0) + 1
        }

    except Exception as e:
        print(f"[Prober] Lỗi khi sinh câu hỏi (Timeout/Format): {e}")
        # Nếu lỗi, cứ tăng iteration_count để không bị lặp vô hạn,
        # và giữ nguyên current_case hoặc set None tùy chiến lược routing ở Đồ thị cha.
        return {"iteration_count": state.get("iteration_count", 0) + 1}