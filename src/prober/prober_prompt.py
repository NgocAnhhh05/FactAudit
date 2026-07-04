# deep_search_prompt = """This task involves generating test cases for the fact-checking task. Fact-checking is an important capability of LLMs, where the LLM should analyze textual information to identify the factuality of the source claim. Here, you need to ask the LLM to be tested to accurately assess the factuality of the information presented with the claim itself, or auxiliary information.

# Previous Prompts (History of Target LLM's Performance):
# {history_context}

# The objective is to create a NEW prompt that is challenging for the language model, with a focus on diverse types of instructions about "{task_name}". Each prompt should be solvable by a language model completely, and aimed at achieving a lower score (indicating difficulty and complexity).

# Guidelines for Creating New Prompts:
# 1. Each prompt to be fact-checked should be solvable by a language model (no visual task) and should contain all necessary information.
# 2. Understand how factuality in the Prompt is discerned by the tested model as shown in its Answer and the comment on the score.
# 3. Aim for prompts that would result in a low score (less than 3.0), indicating a high level of complexity and difficulty of the question that requires more effort to identify the factuality or misinformation.
# 4. Do not repeat topics across different instructions in the previous prompts to maximize diversity.
# 5. The key_point should be no more than 15 words and summarize the key points of the prompt to clearly state the target content to be fact-checked.
# 6. The test_mode should be one of the three options: 1) [claim], (i.e., only the source claim), or 2) [evidence], (i.e., additional contrastive evidence based on Wikipedia), or 3) [wisdom of crowds], (i.e., user comments on social media).
# 7. The auxiliary_info should be provided according to the test_mode: if not the [claim] mode is selected, generate the auxiliary information "auxiliary_info" for the source claim. If else, "auxiliary_info" is empty.
#     - For "auxiliary_info" of [evidence], please ensure that: 1) more than three pieces of evidence are in "auxiliary_info", and 2) the provided pieces of detailed evidence in "auxiliary_info" must only be ground truth quoted directly and solely from Wikipedia word for word (without any personal insight), where different amounts of supported, refuted, and neutral evidence to the source claim should be included.
#     - For "auxiliary_info" of [wisdom of crowds], please ensure that: 1) the depth of the conversation tree in "auxiliary_info" must be more than two, and 2) the hierarchical conversation tree in "auxiliary_info" can be noisy but valuable to help verify the source claim.
# 8. Please focus on "{task_name}" constraints, and ensure that upon careful consideration a human fact-checker with commonsense can identify the factuality of the new prompt.
# 9. The new prompt should be STRICTLY within 512 words and should not be too long.

# Please generate a new test case following the structured output format.
# """

deep_search_prompt = """This task involves generating test cases for the fact-checking task. Fact-checking is an important capability of LLMs, where the LLM should analyze textual information to identify the factuality of the source claim. Here, you need to ask the LLM to be tested to accurately assess the factuality of the information presented with the claim itself, or auxiliary information.

Previous Prompts (History of Target LLM's Performance):
{history_context}

The objective is to create a NEW prompt that is challenging for the language model, with a focus on diverse types of instructions about "{task_name}". Each prompt should be solvable by a language model completely, and aimed at achieving a lower score (indicating difficulty and complexity).

Guidelines for Creating New Prompts:
1. Each prompt to be fact-checked should be solvable by a language model (no visual task) and should contain all necessary information.
2. Understand how factuality in the Prompt is discerned by the tested model as shown in its Answer and the comment on the score.
3. Aim for prompts that would result in a low score (less than 3.0), indicating a high level of complexity and difficulty of the question that requires more effort to identify the factuality or misinformation.
4. Do not repeat topics across different instructions in the previous prompts to maximize diversity.
5. CRITICAL REQUIREMENT FOR 'key_point': The key_point MUST be highly specific, detailed, and unambiguous. It must clearly state the exact claim, context, entities, and figures (if any) being fact-checked. DO NOT write vague or overly broad statements.
   - BAD key_point (Vague): "A claim about a new cure for diabetes."
   - BAD key_point (Broad): "Study finds link between coffee and cancer."
   - GOOD key_point (Specific): "A viral social media post claims that drinking apple cider vinegar daily cures Type 2 Diabetes within 4 weeks."
   - GOOD key_point (Specific): "A recent study claims drinking 3 cups of coffee daily reduces liver cancer risk by 50%."
6. The test_mode should be one of the three options: 1) [claim], (i.e., only the source claim), or 2) [evidence], (i.e., additional contrastive evidence based on Wikipedia), or 3) [wisdom of crowds], (i.e., user comments on social media).
7. The auxiliary_info should be provided according to the test_mode: 
    - For [claim]: Just leave "auxiliary_info" empty.
    - For [evidence]: Provide pieces of detailed evidence in "auxiliary_info". The evidence must be grounded in truth based on Wikipedia or other authorities, where supported, refuted, and neutral evidence to the source claim should be included.
    - For [wisdom of crowds]: Ensure that the depth of the conversation tree in "auxiliary_info" must be more than two, and the hierarchical conversation tree can be noisy but valuable to help verify the source claim.
8. Please focus on "{task_name}" constraints, and ensure that upon careful consideration a human fact-checker with commonsense can identify the factuality of the new prompt.
9. The new prompt should be STRICTLY within 512 words and should not be too long.

Please generate a new test case following the structured output format.
"""