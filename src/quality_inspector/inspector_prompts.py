# Quality Inspector Agent

# judge_new_case_prompt = """Fact-checking is an important capability of LLMs, where the LLM should analyze textual information to identify the factuality of the source claim. Here, the LLM must be tested to accurately assess the factuality of the information presented within the source claim according to the claim itself or the auxiliary information.

# Please judge whether the new test case below is suitable as a diverse and comprehensive exam question on the subtask "{task_name}".

# [TEST CASE TO INSPECT]
# {test_case_json}

# The judgment criteria are as follows:
# 1. The claim should be important and meaningful to the task "{task_name}", avoiding unnecessary ambiguity in the key point.
# 2. If "auxiliary_info" is not empty, it can be noisy but must be helpful to the fact verification process; If "auxiliary_info" is empty, just keep it empty.
# 3. If "test_mode" is [claim], "auxiliary_info" MUST be empty.
# 4. If "test_mode" is [wisdom of crowds], please check "auxiliary_info" that: a) the user comments should be valuable enough as the wisdom of crowds for fact verification and b) the depth of the propagation conversation tree composed of the user response MUST be a random integer more than two.
# 5. If "test_mode" is [evidence], please check "auxiliary_info" that: a) four or more random pieces of evidence are in "auxiliary_info", and b) the provided pieces of detailed evidence must be ONLY ground truth based on Wikipedia or other authority, where all supported, refuted, and neutral evidence to the source claim should be included.
# 6. The fact-checking topic should be diverse and clear.

# If the test case conforms to ALL criteria, set is_valid to True and return the case exactly as it is.
# If it violates ANY criteria, set is_valid to False, provide feedback on what is wrong, and generate a REVISED test case that fixes the issues while strictly following the criteria.
# """

judge_new_case_prompt = """Fact-checking is an important capability of LLMs, where the LLM should analyze textual information to identify the factuality of the source claim. Here, the LLM must be tested to accurately assess the factuality of the information presented within the source claim according to the claim itself or the auxiliary information.

Please judge whether the new test case below is suitable as a diverse and comprehensive exam question on the subtask "{task_name}".

[TEST CASE TO INSPECT]
{test_case_json}

The judgment criteria are as follows:
1. The claim should be important and meaningful to the task "{task_name}". The "key_point" MUST be highly specific, detailed, and unambiguous. Avoid any vague or overly broad statements.
2. If "auxiliary_info" is not empty, it can be noisy but must be helpful to the fact verification process.
3. If "test_mode" is [wisdom of crowds], please check "auxiliary_info" that: a) the user comments should be valuable enough as the wisdom of crowds for fact verification and b) the depth of the propagation conversation tree composed of the user response MUST be a random integer more than two.
4. If "test_mode" is [evidence], please check "auxiliary_info" that: the provided pieces of detailed evidence must be ONLY ground truth based on Wikipedia or other authority, where all supported, refuted, and neutral evidence to the source claim should be included.
5. The fact-checking topic should be diverse and clear.

If the test case conforms to ALL criteria, set is_valid to True and return the case exactly as it is.
If it violates ANY criteria, set is_valid to False, provide feedback on what is wrong, and generate a REVISED test case that fixes the issues while strictly following the criteria.
"""


verification_prompt = """You are a strict, objective fact-checker.
We have an AI-generated test case containing a claim and some evidence. We need to verify if the generated evidence is based on reality or if it is hallucinated.

[Claim to check]: 
{claim}

[AI-Generated Evidence]: 
{evidence}

[Real-time Web Search Results]:
{web_results}

Task:
Compare the [AI-Generated Evidence] against the [Real-time Web Search Results].
1. If the generated evidence is supported by the web results, set is_factual to True.
2. If the generated evidence is hallucinated, fabricated, or contradicts the real facts in the web results, set is_factual to False.
3. If False, write the correct facts found in the search results into the 'correction' field so the evidence can be rewritten later.
"""