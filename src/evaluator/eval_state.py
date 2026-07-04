"""
State Definitions and Structured Outputs for Evaluator Agent

Updated to perfectly match FACT-AUDIT's original 1-10 scoring system,
strict classification labels, and the voting/refining pipeline.
"""

from typing import Literal, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

class Phase1State(TypedDict):
    """
    Internal State for Phase 1 (Self-Consistency Sub-graph).
    Isolates the parallel answers and the refinement process.
    """

    current_case: dict
    ref_ans_1: Optional[str] 
    ref_ans_2: Optional[str]
    ref_ans_3: Optional[str]
    voted_answer: Optional[str]     # Kết quả sau khi gộp 3 luồng
    refined_answer: Optional[str]   # Kết quả cuối cùng sau khi Refine làm mịn


FactLabel = Literal["Factual", "Non-Factual", "Not Enough Information"]

class ReferenceOutput(BaseModel):
    verdict: FactLabel = Field(description="The final verdict for the claim.")
    justification: str = Field(description="Detailed factual analysis and reasoning.")


class VoteOutput(BaseModel):
    verdict: FactLabel = Field(description="The merged verdict from the three answers.")
    justification: str = Field(description="The summarized and corrected justification.")


class RefineOutput(BaseModel):
    refined_answer: str = Field(
        description="The final, polished reference answer containing both the verdict and the justification."
    )


class ScoreOutput(BaseModel):
    score: float = Field(
        description="A score from 1.0 to 10.0 indicating the quality of the Target LLM's answer."
    )
    comparison: str = Field(
        description="Explanation comparing the target response with the reference response."
    )