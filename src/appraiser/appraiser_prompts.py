analysis_prompt = """Fact-checking is an important capability of LLMs, where the LLM should analyze textual information to identify the factuality of the source claim. Here, the LLM must be tested to accurately assess the factuality of the information presented within the source claim according to the claim itself or the auxiliary information.

Here is a sub task's taxonomy as well as the averaged score on these tasks(lower means worse performance):
{taxonomy}

And here are some bad cases:
{bad_cases}
Based on the given information, please judge if the taxonomy is comprehensive, if so please just output [[Stop]]. 

If not, please give me a new possible issue you inferred from the present taxonomy and bad cases. Please focus on {main_task}. Ensure the new task is text-only (no multimodal). Also give a brief explanation of how you find the issue. Please output in a JSON format: {{"task_name": "...", "explanation": "..."}}"""


judge_new_task_prompt = """Fact-checking is an important capability of LLMs, where the LLM should analyze textual information to identify the factuality of the source claim. Here, the LLM must be tested to accurately assess the factuality of the information presented within the source claim according to the claim itself or the auxiliary information.

Here is a sub task's taxonomy on the task "{main_task}":
{taxonomy}

Based on the given taxonomy, please judge whether the new test point "{new_point}" is suitable as a sub task on the task "{main_task}". The judge criteria are as follows:
1. The new test point should precisely cover an important and meaningful part of the main task.
2. The new test point should be sufficiently different from the existing test points.
3. The new test point should be text-only (no multimodal).

If the new test point "{new_point}" is suitable as a sub task on the task "{main_task}", please ONLY output [[Yes]]. If not, please first output [[No]], and then provide the reason why it's not suitable as a sub task on the task "{main_task}"."""