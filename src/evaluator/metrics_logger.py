import json
import os
from filelock import FileLock

def log_case_to_json(state_record: dict, file_path: str = "memory_pool.json"):
    """
    Hàm lưu record đã cấu trúc vào file JSON, sử dụng FileLock
    để ngăn chặn Race Condition khi chạy đa luồng (Fan-out).
    """
    prompt_raw = state_record.get("prompt", {})
    ref_ans = state_record.get("reference_answer", "")

    gold_verdict = "Not Enough Information"
    if ref_ans.startswith("[Factual]"):
        gold_verdict = "Factual"
    elif ref_ans.startswith("[Non-Factual]"):
        gold_verdict = "Non-Factual"

    clean_record = {
        "key_point": state_record.get("key_point", ""),
        "test_mode": state_record.get("test_mode", ""),
        "target_llm_verdict": "Factual" if "Verdict: Factual" in state_record.get("target_response", "") else ("Non-Factual" if "Verdict: Non-Factual" in state_record.get("target_response", "") else "Not Enough Information"),
        "gold_verdict": gold_verdict,
        "grade": float(state_record.get("score", 0.0))
    }

    lock_path = file_path + ".lock"
    lock = FileLock(lock_path, timeout=10) # Chờ tối đa 10 giây nếu file đang bị luồng khác khóa

    with lock:
        data = []
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = []

        data.append(clean_record)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)