"""
State Definitions and Pydantic Schemas for Appraiser Agent

This module defines the core state objects and structured output schemas
used in the Appraiser agent's workflow. It includes the runtime state for
managing the taxonomy update process, as well as structured schemas to enforce
consistent LLM outputs for taxonomy analysis and new task judgment.
"""

from typing import Dict, TypedDict, Optional
from pydantic import BaseModel, Field

class AppraiserState(TypedDict):
    """
    Internal State for the Appraiser sub-graph.
    
    Manages the inputs from the main graph, internal variables for tracking
    retries and temporary proposals, and outputs to return to the parent graph.
    """

    # Input from Main Graph
    main_task: str
    taxonomy_scores: Dict[str, float]
    bad_cases_formatted: str

    # Internal State for subgraph execution
    current_new_task: Optional[str]
    current_explanation: Optional[str]
    retry_count: int

    # Output to Main Graph
    final_new_task: Optional[str]
    is_terminated: bool


class AnalysisOutput(BaseModel):
    """
    Schema for analyzing the current taxonomy and proposing a new task.

    Forces LLMs to return structured output when evaluating the taxonomy's comprehensiveness.
    - is_stop: 'True' if no new test scenarios are needed, 'False' otherwise.
    - task_name: The proposed scenario name to target the model's weakness.
    - explanation: Why this specific scenario was chosen based on the bad cases.
    """

    is_stop: bool = Field(
        description="True if the taxonomy is comprehensive. False otherwise."
        )
    
    task_name: str = Field(default="", 
                           description="The name of the new test scenario inferred from bad cases."
                           )

    explanation: str = Field(default="", 
                             description="A brief explanation of how you found the issue."
                             )

class JudgeOutput(BaseModel):
    """
    Schema for judging the suitability of a newly proposed task.

    Forces LLMs to return structured output indicating if the new task meets all criteria
    (distinctiveness, relevance, and format).
    - is_suitable: 'True' if the task is accepted, 'False' if it violates criteria.
    - reason: Explanation for rejection (required if is_suitable is False).
    """
    is_suitable: bool = Field(
        description="True if the new test point is suitable. False if not."
        )
    
    reason: str = Field(default="", 
                        description="Provide the reason if is_suitable is False."
                        )