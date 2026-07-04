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