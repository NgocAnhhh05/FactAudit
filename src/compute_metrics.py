import os
import pandas as pd

def calculate_fact_audit_metrics(file_path: str = "memory_pool.json"):
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        print("Hệ thống kiểm toán rỗng hoặc chưa có dữ liệu trong Memory Pool!")
        return

    df = pd.read_json(file_path)
    total_cases = len(df)

    #1: Grade
    avg_grade = df["grade"].mean()

    #2: IMR
    imr = (df["grade"] <= 3.0).sum() / total_cases * 100

    #3: JFR
    # Condition: target_llm_verdict == gold_verdict
    correct_verdict_mask = df["target_llm_verdict"] == df["gold_verdict"]
    flawed_justification_mask = df["grade"] <= 3.0

    jfr = (correct_verdict_mask & flawed_justification_mask).sum() / total_cases * 100

    print("\n" + "="*50)
    print("FACT-AUDIT METRICS")
    print("="*50)
    print(f"Total questions: {total_cases}")
    print(f"Grade: {avg_grade:.2f} / 10.0")
    print(f"IMR (Insight Mastery Rate): {imr:.2f}%")
    print(f"JFR (Justification Flaw Rate): {jfr:.2f}%")
    print("="*50 + "\n")

if __name__ == "__main__":
    calculate_fact_audit_metrics()