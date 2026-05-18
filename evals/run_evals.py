import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../src"))

import csv
from agent import build_graph

def run_evals(test_file: str) -> None:
    app = build_graph()
    results = []
    passed = 0

    with open(test_file, "r") as f:
        reader = csv.DictReader(f)
        test_cases = list(reader)

    print(f"Running {len(test_cases)} test cases...\n")

    for i, case in enumerate(test_cases):
        question = case["question"]
        keywords = case["expected_keywords"].split("|")

        result = app.invoke({"query": question, "context": "", "answer": ""})
        answer = result["answer"].lower()

        matched = [k for k in keywords if k.lower() in answer]
        score = len(matched) / len(keywords)
        status = "✅ PASS" if score >= 0.5 else "❌ FAIL"

        if score >= 0.5:
            passed += 1

        print(f"Q{i+1}: {question}")
        print(f"Status: {status} (matched {len(matched)}/{len(keywords)} keywords)")
        print(f"Answer: {answer[:150]}...")
        print()

        results.append({
            "question": question,
            "status": status,
            "score": score,
            "matched_keywords": ", ".join(matched),
            "answer": answer[:300]
        })

    accuracy = passed / len(test_cases) * 100
    print(f"Final accuracy: {passed}/{len(test_cases)} ({accuracy:.1f}%)")

    with open("evals/results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "status", "score", "matched_keywords", "answer"])
        writer.writeheader()
        writer.writerows(results)

    print("\nResults saved to evals/results.csv")


if __name__ == "__main__":
    run_evals("evals/test_cases.csv")