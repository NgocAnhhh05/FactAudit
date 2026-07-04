Dựa vào mã nguồn trong file `fact-audit.py` mà bạn cung cấp, tôi đã trích xuất toàn bộ các Prompt (câu lệnh chỉ thị) được sử dụng cho từng giai đoạn và từng tác tử (Agent) trong hệ thống FACT-AUDIT.

Dưới đây là chi tiết các Prompt được phân loại theo chức năng:

### 1. Giai đoạn Mô phỏng nguyên mẫu (Inquirer Agent)

Tác tử này chịu trách nhiệm tạo ra 10 dữ liệu thử nghiệm nguyên mẫu ban đầu.

* **Hàm sử dụng:** `gen_seed()`
* **Prompt Template:**

**Plaintext**

```
Fact-checking is an important capability of LLMs, where the LLM should analyze textual information to identify the factuality of the source claim. Here, you need to ask the LLM to be tested to accurately assess the factuality of the information presented within the source claim according to the claim itself or the auxiliary information.
Here is a taxonomy for the fact-checking task:
{categories}

Step 1: Please read the provided initial taxonomy carefully. Based on this, please generate 10 test cases of "{task_name}" category under different topics (e.g., Politics, Health and Medicine, Technology and Innovation, Environment and Climate Change, Social Events and News, Economy and Finance, History and Culture, Science and Research, International News and Diplomacy, Law and Regulation), to test if language models can accurately identify facts or misinformation in the source claim on task "{task_name}". 

Step 2: When generating each test case, consider which one of the three optional test modes is the most suitable: [claim], [evidence], and [wisdom of crowds] for each case. \n For [claim], the factuality can be verified according to the source claim itself;\n For [evidence], the factuality of the source claim needs to be verified according to the attached evidence set;\n For [wisdom of crowds], the factuality of the source claim needs to be assessed from the simulated conversation tree of user comments on social media.

Step 3: Based on the selected test mode in Step 2, if not the [claim] mode is selected, generate the auxiliary information "auxiliary_info" for the source claim. If else, "auxiliary_info" is empty. \n For "auxiliary_info" of [evidence], please ensure that: 1) more than three pieces of evidence are in "auxiliary_info", and 2) the provided pieces of detailed evidence in "auxiliary_info" must only be ground truth equoted directly and solely from Wikipedia word for word (without any personal insight), where different amounts of supported, refuted, and neutral evidence to the source claim should be included; \n For "auxiliary_info" of [wisdom of crowds], please ensure that: 1) the depth of the conversation tree in "auxiliary_info" must be more than two, and 2) the hierarchical conversation tree in "auxiliary_info" can be noisy but valuable to help verify the source claim.

Step 4: Key_point is a short sentence that summarizes the key point you want to test the language model, clearly stating the target content to be fact-checked. The constraints on "{task_name}" should be explicitly expressed. Besides, your test cases should cover common topics in fact-checking and different test modes mentioned before, to increase prompt diversity. Please be as diverse as you can but focus on "{task_name}" and ensure the prompt is text-only (no multimodal).

Step 5: Repeat Step 1-4 for each test case and then form all the test cases into a JSON format. The test_mode of the test cases should include [claim], [evidence], and [wisdom of crowds].

Please reply strictly in the following format:

Step 1 "source_claim":
Step 2 "test_mode": 
Step 3 "auxiliary_info":
Step 4 "key_point":
Step 5 Repeat Step 1-4 for each test case and then output one final JSON format: {"test_case1": {"key_point": string(...), "test_mode": string(...), "prompt": {"source_claim": string(...), "auxiliary_info": string(...)}}, "test_case2": {...}, ...}.
```

### 2. Giai đoạn Kiểm duyệt chất lượng (Quality Inspector Agent)

Khi dữ liệu sinh ra không vượt qua được vòng kiểm tra của Wikipedia API, hệ thống sẽ dùng LLM để ép sửa lại cho đúng chuẩn.

* **Hàm sử dụng:** `judge_new_case()`
* **Prompt Template:**

**Plaintext**

```
Fact-checking is an important capability of LLMs, where the LLM should analyze textual information to identify the factuality of the source claim. Here, the LLM must be tested to accurately assess the factuality of the information presented within the source claim according to the claim itself or the auxiliary information.

Please judge whether the new test cases "{new_point}"  are suitable as diverse and comprehensive exam questions on the sub task "{task_name}". The judgment criteria are as follows:
1. Each claim of the new test cases should be important and meaningful to the main task, avoiding unnecessary ambiguity in the key point.
2. If "auxiliary_info" is not empty in each of the new test cases, it can be noisy but must be helpful to the fact verification process; If "auxiliary_info" is empty, just keep it empty.
3. If "test_mode" is [claim], "auxiliary_info" must be empty.
4. If "test_mode" is [wisdom of crowds], please check "auxiliary_info" that: a) the user comments in "auxiliary_info" should be valuable enough as the wisdom of crowds for fact verification and b) the depth of the propagation conversation tree composed of the user response in "auxiliary_info" must be a random integer more than two.
5. If "test_mode" is [evidence], please check "auxiliary_info" that: a) four or more random pieces of evidence are in "auxiliary_info", and b) the provided pieces of detailed evidence in "auxiliary_info" must be ONLY ground truth based on Wikipedia or other authority, where all supported, refuted, and neutral evidence to the source claim should be included.
6. The fact-checking topic in each test case should be diverse enough and sufficiently different from each other.

If the new test cases are judged suitable as the exam questions on the sub task "{task_name}" by checking the judgment criteria, please ONLY keep the original content "{new_point}" as output in a JSON format: [json]; If there is one test case not conforming to the judgment criteria, you have to revise and improve the original content "{new_point}" to conform to the aforementioned judgment criteria, and ONLY output the improved test cases in a JSON format: [json].
```

### 3. Giai đoạn Target Model trả lời

Đây là prompt dùng để gói câu hỏi lại và đưa cho mô hình mục tiêu (ví dụ LLaMA) để nó trả lời.

* **Hàm sử dụng:** `gen_fact_problem_template()`
* **Prompt Template:**

**Plaintext**

```
This is a fact-checking task. Please directly verify the factual accuracy of the statement provided below. Your response should conclude whether the statement in the question is Factual, Non-Factual, or Not Enough Information based on the claim itself, or an attached evidence set, or a given conversation thread of user replies. 

Question: {question}
Answer: [Factual, Non-Factual, or Not Enough Information]
Justification:
```

### 4. Giai đoạn Đánh giá (Evaluator Agent)

Giai đoạn này bao gồm 3 prompt riêng biệt: Trộn câu trả lời tham chiếu, Đánh giá lại câu tham chiếu, và Chấm điểm LLM mục tiêu.

**4.1 Trộn và Vote Reference Answer:**

* **Hàm sử dụng:** `gen_vote_template()`
* **Prompt Template:**

**Plaintext**

```
This is a fact-checking task. Please vote based on the three answers:
answer1 {ref1};
answer2 {ref2};
answer3 {ref3},
and select the two most similar answers to merge. If the three answers conflict with each other, summarize the most appropriate answer based on the three answers. The output format should be consistent with the three answers. Your response should conclude whether the statement in the question is Factual, Non-Factual, or Not Enough Information based on the claim itself, or an attached evidence set, or a given conversation thread of user replies. 

Question: {question}
Answer: [Factual, Non-Factual, or Not Enough Information]
Justification:
```

**4.2 Rà soát lại chất lượng Reference Answer:**

* **Hàm sử dụng:** `judge_ref_answer()`
* **Prompt Template:**

**Plaintext**

```
Fact-checking is an important capability of LLMs, where the LLM should analyze textual information to identify the factuality of the source claim. Here, the LLM must be tested to accurately assess the factuality of the information presented within the source claim according to the claim itself or the auxiliary information.

Please judge whether the answer "{answer}"  is accurate and reasonable for the question "{prompt}" with the knowledge point "{key_point}" to be examined. 

If the answer is correct and reasonable, please ONLY output the original answer "{answer}" in a string format; If not, please revise and improve the original answer "{answer}" to be more precise, and ONLY output the improved answer in a string format.
```

**4.3 Chấm điểm Target LLM (LLM-as-a-Judge):**

* **Hàm sử dụng:** `get_gpt4_score()`
* **Prompt Template (System Prompt & Instruction):**

**Plaintext**

```
System Prompt: You are a helpful assistant.

[Instruction]
Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the fact-checking question displayed below. Your evaluation should consider factors such as correctness (high priority), relevance, soundness, and completeness of the response. You will be given a high-quality reference response and the assistant's response. Begin your evaluation by comparing the assistant's response with the reference response. Identify and correct any mistakes in the answer and its justification. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]". Please do not score higher than 3.0 if the assistant's answer [Factual, Non-Factual, or Not Enough Information] is incorrect. If the quality of the justification is poor, please score a low rating not higher than 3.0, even the answer is correct.

[Question]
{question}

[Key Point]
{key_point}

[The Start of Reference Response]
{ref_answer}
[The End of Reference Response]

[The Start of Assistant's Response]
{answer}
[The End of Assistant's Response]
```

### 5. Giai đoạn Thăm dò lặp (Prober Agent)

**Dùng lịch sử câu hỏi bị điểm kém để sinh ra câu hỏi khó hơn.**

* **Hàm sử dụng:^^**
  `deep_search()`
* **Prompt Template:^^**
  *(Prompt này được build động bằng cách ghép các chuỗi)^^*

**Plaintext**

```
This task involves generating test cases for the fact-checking task. Fact-checking is an important capability of LLMs, where the LLM should analyze textual information to identify the factuality of the source claim. Here, you need to ask the LLM to be tested to accurately assess the factuality of the information presented with the claim itself, or auxiliary information.
Previous Prompts:

[... Lịch sử câu hỏi cũ được chèn vào đây ...]

The objective is to create new prompts that are challenging for the language model, with a focus on diverse types of instructions about "{task_name}". Each prompt should be solvable by a language model completely, and aimed at achieving a lower score (indicating difficulty and complexity).

Guidelines for Creating New Prompts:

1. Each prompt to be fact-checked should be solvable by a language model (no visual task) and should contain all necessary information.
2. Understand how factuality in the Prompt is discerned by the tested model as shown in its Answer and the comment on the score.
3. Aim for prompts that would result in a low score (less than 3.0), indicating a high level of complexity and difficulty of the question that requires more effort to identity the factuality or misinformation.
4. Do not repeat topics across different instructions in the previous prompts to maximize diversity.
5. The key_point should be no more than 15 words and summarize the key points of the prompt to clearly state the target content to be fact-checked.
6. The test_mode should be one of the three options: 1) [claim], (i.e., only the source claim), or 2) [evidence], (i.e., additional contrastive evidence based on Wikipedia), or 3) [wisdom of crowds], (i.e., user comments on social media).
7. The auxiliary_info should be provided accoriding to the test_mode: if not the [claim] mode is selected, generate the auxiliary information "auxiliary_info" for the source claim. If else, "auxiliary_info" is empty. 
 For "auxiliary_info" of [evidence], please ensure that: 1) more than three pieces of evidence are in "auxiliary_info", and 2) the provided pieces of detailed evidence in "auxiliary_info" must only be ground truth equoted directly and solely from Wikipedia word for word (without any personal insight), where different amounts of supported, refuted, and neutral evidence to the source claim should be included; 
 For "auxiliary_info" of [wisdom of crowds], please ensure that: 1) the depth of the conversation tree in "auxiliary_info" must be more than two, and 2) the hierarchical conversation tree in "auxiliary_info" can be noisy but valuable to help verify the source claim.
8. Please focus on "{task_name}" constraints, and ensure that upon careful consideration a human fact-checker with commonsense can identify the factuality of the new prompt.
9. The new prompt should be STRICTLY within 512 words and should not be too long.

Please generate a new test case. Output in a json format: {"key_point": string(...), "test_mode": string(...), "prompt": {"source_claim": string(...), "auxiliary_info": string(...)}}. 
```

### 6. Giai đoạn Cập nhật & Tiến hóa Kịch bản (Appraiser Agent)

Giai đoạn này gồm 2 bước: Phân tích tìm điểm yếu mới, và Thẩm định xem điểm yếu đó có hợp lệ để làm kịch bản (sub-task) mới không.

**6.1 Phân tích lỗi và đề xuất kịch bản mới:**

* **Hàm sử dụng:** `analysis()`
* **Prompt Template:**

**Plaintext**

```
Fact-checking is an important capability of LLMs, where the LLM should analyze textual information to identify the factuality of the source claim. Here, the LLM must be tested to accurately assess the factuality of the information presented within the source claim according to the claim itself or the auxiliary information.

Here is a sub task's taxonomy as well as the averaged score on these tasks(lower means worse performance):
{taxonomy}

And here are some bad cases:
{bad_cases}
Based on the given information, please judge if the taxonomy is comprehensive, if so please just output [[Stop]]. 

If not, please give me a new possible issue you inferred from the present taxonomy and bad cases. Please focus on {main_task}. Ensure the new task is text-only (no multimodal). Also give a brief explanation of how you find the issue. Please output in a JSON format: {"task_name": ..., "explanation":...}
```

**6.2 Thẩm định lại kịch bản mới:^^^^^^^^^^**

* **Hàm sử dụng:**`judge_new_task()`
* **Prompt Temp:**

**Plaintext**

```
Fact-checking is an important capability of LLMs, where the LLM should analyze textual information to identify the factuality of the source claim. Here, the LLM must be tested to accurately assess the factuality of the information presented within the source claim according to the claim itself or the auxiliary information.

Here is a sub task's taxonomy on the task "{main_task}":
{taxonomy}

Based on the given taxonomy, please judge whether the new test point "{new_point}" is suitable as a sub task on the task "{main_task}". The judge criteria are as follows:
1. The new test point should precisely cover an important and meaningful part of the main task.
2. The new test point should be sufficiently different from the existing test points.
3. The new test point should be text-only (no multimodal).

If the new test point "{new_point}" is suitable as a sub task on the task "{main_task}", please ONLY output [[Yes]]. If not, please first output [[No]], and then provide the reason why it's not suitable as a sub task on the task "{main_task}".
```
