# gen_seed_prompt = """Fact-checking is an important capability of LLMs, where the LLM should analyze textual information to identify the factuality of the source claim. Here, you need to ask the LLM to be tested to accurately assess the factuality of the information presented within the source claim according to the claim itself or the auxiliary information.
# Here is a taxonomy for the fact-checking task:
# {categories}

# Step 1: Please read the provided initial taxonomy carefully. Based on this, please generate 10 test cases of "{task_name}" category under different topics (e.g., Politics, Health and Medicine, Technology and Innovation, Environment and Climate Change, Social Events and News, Economy and Finance, History and Culture, Science and Research, International News and Diplomacy, Law and Regulation), to test if language models can accurately identify facts or misinformation in the source claim on task "{task_name}". 

# Step 2: When generating each test case, consider which one of the three optional test modes is the most suitable: [claim], [evidence], and [wisdom of crowds] for each case. \n For [claim], the factuality can be verified according to the source claim itself;\n For [evidence], the factuality of the source claim needs to be verified according to the attached evidence set;\n For [wisdom of crowds], the factuality of the source claim needs to be assessed from the simulated conversation tree of user comments on social media.

# Step 3: Based on the selected test mode in Step 2, if not the [claim] mode is selected, generate the auxiliary information "auxiliary_info" for the source claim. If else, "auxiliary_info" is empty. \n For "auxiliary_info" of [evidence], please ensure that: 1) more than three pieces of evidence are in "auxiliary_info", and 2) the provided pieces of detailed evidence in "auxiliary_info" must only be ground truth equoted directly and solely from Wikipedia word for word (without any personal insight), where different amounts of supported, refuted, and neutral evidence to the source claim should be included; \n For "auxiliary_info" of [wisdom of crowds], please ensure that: 1) the depth of the conversation tree in "auxiliary_info" must be more than two, and 2) the hierarchical conversation tree in "auxiliary_info" can be noisy but valuable to help verify the source claim.

# Step 4: Key_point is a short sentence that summarizes the key point you want to test the language model, clearly stating the target content to be fact-checked. The constraints on "{task_name}" should be explicitly expressed. Besides, your test cases should cover common topics in fact-checking and different test modes mentioned before, to increase prompt diversity. Please be as diverse as you can but focus on "{task_name}" and ensure the prompt is text-only (no multimodal).

# Step 5: Repeat Step 1-4 for each test case and then form all the test cases into a JSON format. The test_mode of the test cases should include [claim], [evidence], and [wisdom of crowds].

# Please reply strictly in the following format:

# Step 1 "source_claim":
# Step 2 "test_mode": 
# Step 3 "auxiliary_info":
# Step 4 "key_point":
# Step 5 Repeat Step 1-4 for each test case and then output one final JSON format: 
# {{
#     "test_case1": {{
#         "key_point": "string(...)", 
#         "test_mode": "string(...)", 
#         "prompt": {{
#             "source_claim": "string(...)", 
#             "auxiliary_info": "string(...)"
#         }}
#     }}, 
#     "test_case2": {{...}}, 
#     ...
# }}"""

gen_seed_prompt = """Fact-checking is an important capability of LLMs, where the LLM should analyze textual information to identify the factuality of the source claim. Here, you need to ask the LLM to be tested to accurately assess the factuality of the information presented within the source claim according to the claim itself or the auxiliary information.
Here is a taxonomy for the fact-checking task:
{categories}

Step 1: Please read the provided initial taxonomy carefully. Based on this, please generate 10 test cases of "{task_name}" category under different topics (e.g., Politics, Health and Medicine, Technology and Innovation, Environment and Climate Change, Social Events and News, Economy and Finance, History and Culture, Science and Research, International News and Diplomacy, Law and Regulation), to test if language models can accurately identify facts or misinformation in the source claim on task "{task_name}". 

Step 2: When generating each test case, consider which one of the three optional test modes is the most suitable: [claim], [evidence], and [wisdom of crowds] for each case. 
- For [claim], the factuality can be verified according to the source claim itself.
- For [evidence], the factuality of the source claim needs to be verified according to the attached evidence set.
- For [wisdom of crowds], the factuality of the source claim needs to be assessed from the simulated conversation tree of user comments on social media.

Step 3: Generate the auxiliary information "auxiliary_info" based on the selected test mode:
- For [claim]: Leave "auxiliary_info" empty.
- For [evidence]: Provide pieces of detailed evidence in "auxiliary_info". The evidence must be grounded in truth, quoted directly and solely from Wikipedia or other authorities word for word (without personal insight), where supported, refuted, and neutral evidence to the source claim should be included.
- For [wisdom of crowds]: Ensure that the depth of the conversation tree in "auxiliary_info" must be more than two, and the hierarchical conversation tree can be noisy but valuable to help verify the source claim.

Step 4: CRITICAL REQUIREMENT FOR 'key_point': Key_point is a short sentence that summarizes the exact point you want to test the language model. It MUST be highly specific, detailed, and unambiguous. DO NOT write vague or overly broad statements.
   - BAD key_point (Vague): "A claim about a new cure for diabetes."
   - BAD key_point (Broad): "Study finds link between coffee and cancer."
   - GOOD key_point (Specific): "A viral social media post claims that drinking apple cider vinegar daily cures Type 2 Diabetes within 4 weeks."
   - GOOD key_point (Specific): "A recent study claims drinking 3 cups of coffee daily reduces liver cancer risk by 50%."
The constraints on "{task_name}" should be explicitly expressed. Your test cases should cover common topics and different test modes. Ensure the prompt is text-only.

Step 5: Form all the test cases into the structured output format. Ensure the test modes are diverse among the 10 cases.
"""