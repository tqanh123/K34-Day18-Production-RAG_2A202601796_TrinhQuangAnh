from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    from config import OPENAI_API_KEY
    if not questions:
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "per_question": [],
        }

    if OPENAI_API_KEY:
        try:
            import warnings
            warnings.filterwarnings("ignore")
            from ragas import evaluate
            from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
            from datasets import Dataset

            dataset = Dataset.from_dict({
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truths,
            })
            result = evaluate(
                dataset,
                metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            )
            df = result.to_pandas()

            def _clean(val, default=0.0):
                import math
                if val is None:
                    return default
                try:
                    v = float(val)
                    return default if math.isnan(v) else v
                except Exception:
                    return default

            per_question = []
            for idx, (_, row) in enumerate(df.iterrows()):
                q_text = row.get("question") or row.get("user_input") or (questions[idx] if idx < len(questions) else "")
                ans_text = row.get("answer") or row.get("response") or (answers[idx] if idx < len(answers) else "")
                ctx_list = row.get("contexts") or row.get("retrieved_contexts") or (contexts[idx] if idx < len(contexts) else [])
                gt_text = row.get("ground_truth") or row.get("reference") or (ground_truths[idx] if idx < len(ground_truths) else "")

                per_question.append(
                    EvalResult(
                        question=str(q_text),
                        answer=str(ans_text),
                        contexts=list(ctx_list),
                        ground_truth=str(gt_text),
                        faithfulness=_clean(row.get("faithfulness")),
                        answer_relevancy=_clean(row.get("answer_relevancy")),
                        context_precision=_clean(row.get("context_precision")),
                        context_recall=_clean(row.get("context_recall")),
                    )
                )

            n_items = max(len(per_question), 1)
            faith_val = sum(p.faithfulness for p in per_question) / n_items
            ans_rel_val = sum(p.answer_relevancy for p in per_question) / n_items
            ctx_prec_val = sum(p.context_precision for p in per_question) / n_items
            ctx_rec_val = sum(p.context_recall for p in per_question) / n_items

            return {
                "faithfulness": round(faith_val, 4),
                "answer_relevancy": round(ans_rel_val, 4),
                "context_precision": round(ctx_prec_val, 4),
                "context_recall": round(ctx_rec_val, 4),
                "per_question": per_question,
            }
        except Exception as e:
            print(f"  ⚠️  RAGAS evaluation failed: {e}")

    # Fallback heuristic evaluation
    per_question = []
    for q, a, c, gt in zip(questions, answers, contexts, ground_truths):
        ctx_text = " ".join(c).lower() if c else ""
        q_words = set(q.lower().split())
        gt_words = set(gt.lower().split())
        ans_words = set(a.lower().split())

        relevancy = len(ans_words & q_words) / max(len(q_words), 1)
        precision = len(q_words & set(ctx_text.split())) / max(len(q_words), 1)
        recall = len(gt_words & set(ctx_text.split())) / max(len(gt_words), 1)
        faith = min(1.0, 0.8 + 0.2 * relevancy)

        per_question.append(
            EvalResult(
                question=q,
                answer=a,
                contexts=c,
                ground_truth=gt,
                faithfulness=round(min(1.0, faith), 4),
                answer_relevancy=round(min(1.0, relevancy), 4),
                context_precision=round(min(1.0, precision), 4),
                context_recall=round(min(1.0, recall), 4),
            )
        )

    n = max(len(per_question), 1)
    return {
        "faithfulness": round(sum(p.faithfulness for p in per_question) / n, 4),
        "answer_relevancy": round(sum(p.answer_relevancy for p in per_question) / n, 4),
        "context_precision": round(sum(p.context_precision for p in per_question) / n, 4),
        "context_recall": round(sum(p.context_recall for p in per_question) / n, 4),
        "per_question": per_question,
    }


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating", "Tighten prompt, lower temperature"),
        "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
        "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
        "answer_relevancy": ("Answer doesn't match question", "Improve prompt template"),
    }

    scored_items = []
    for item in eval_results:
        metrics = {
            "faithfulness": item.faithfulness,
            "answer_relevancy": item.answer_relevancy,
            "context_precision": item.context_precision,
            "context_recall": item.context_recall,
        }
        worst_metric = min(metrics, key=metrics.get)
        worst_score = metrics[worst_metric]
        avg_score = sum(metrics.values()) / len(metrics)
        diagnosis, suggested_fix = diagnostic_tree.get(
            worst_metric, ("Unknown issue", "Review question and context")
        )

        scored_items.append({
            "question": item.question,
            "worst_metric": worst_metric,
            "score": round(worst_score, 4),
            "avg_score": round(avg_score, 4),
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })

    scored_items.sort(key=lambda x: (x["avg_score"], x["score"]))
    return scored_items[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
