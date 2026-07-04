gen_fact_problem_prompt = """This is a fact-checking task. Please directly verify the factual accuracy of the statement provided below. Your response should conclude whether the statement in the question is Factual, Non-Factual, or Not Enough Information based on the claim itself, or an attached evidence set, or a given conversation thread of user replies. \n\nQuestion: {question}\nAnswer: [Factual, Non-Factual, or Not Enough Information]\nJustification:"""    
 

# ==========================================
# 1. PROMPT VOTE (Gộp 3 luồng song song)
# ==========================================
gen_vote_prompt = """This is a fact-checking task. Please vote based on the three answers:
answer1: {ref_1};
answer2: {ref_2};
answer3: {ref_3},
and select the two most similar answers to merge. If the three answers conflict with each other, summarize the most appropriate answer based on the three answers. 

The output format should be consistent with the three answers. Your response should conclude whether the statement in the question is Factual, Non-Factual, or Not Enough Information based on the claim itself, or an attached evidence set, or a given conversation thread of user replies.

Question (Claim & Context): {question}

Please extract your final verdict and justification into the structured JSON format provided.
Ensure the answer is exactly one of: [Factual, Non-Factual, Not Enough Information].
"""

# ==========================================
# 2. PROMPT JUDGE/REFINE (Làm mịn đáp án sau khi Vote)
# ==========================================
judge_ref_answer_prompt = """Fact-checking is an important capability of LLMs, where the LLM should analyze textual information to identify the factuality of the source claim. Here, the LLM must be tested to accurately assess the factuality of the information presented within the source claim according to the claim itself or the auxiliary information.

Please judge whether the answer "{answer}" is accurate and reasonable for the question "{prompt}" with the knowledge point "{key_point}" to be examined. 

If the answer is correct and reasonable, please ONLY output the original answer "{answer}" in a string format; If not, please revise and improve the original answer "{answer}" to be more precise, and ONLY output the improved answer.
"""

# ==========================================
# 3. PROMPT SCORING (Chấm điểm Target LLM)
# ==========================================
get_llm_score_prompt = """[Instruction]
Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the fact-checking question displayed below. Your evaluation should consider factors such as correctness (high priority), relevance, soundness, and completeness of the response. 

You will be given a high-quality reference response and the assistant's response. Begin your evaluation by comparing the assistant's response with the reference response. Identify and correct any mistakes in the answer and its justification. Be as objective as possible. 

After providing your explanation, you must rate the response on a scale of 1 to 10. 
CRITICAL RULES: 
- Please do not score higher than 3.0 if the assistant's answer [Factual, Non-Factual, or Not Enough Information] is incorrect. 
- If the quality of the justification is poor, please score a low rating not higher than 3.0, even if the answer is correct.

[Question]
{question}

[Key Point]
{key_point}

[The Start of Reference Response]
{ref_answer}
[The End of Reference Response]

[The Start of Assistant's Response]
{target_response}
[The End of Assistant's Response]
"""