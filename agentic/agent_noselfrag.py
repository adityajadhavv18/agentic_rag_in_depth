"""Offline check: what if the Self-RAG layer's candidates + critic + retry were removed, keeping its "are you sure?"
step on the none route?

Without them the answer is CRAG's own answer: the candidate Self-RAG calls A_all (every kept passage, the usual answer
prompt), which every saved run already holds. So no pipeline re-runs: for each question where Self-RAG picked
something other than A_all, A_all's answer is graded with the same 1-vote and 3-vote graders (cached when an answer
was graded before). Questions without candidates (the none route answered from memory) keep their answer.

Runs compared: GraphRAG FINAL (Phase 1 replay), the agent without web and with web (Phase 3b).
Results: agentic/results/phase3c_no_selfrag_offline.json.

Run from agentic_rag_in_depth/ (OpenAI key sourced; only grader calls):
    uv run agentic/agent_noselfrag.py
"""

import json

import agentic_bridge as ab
from eval_harness import load_golden
from eval_harness.metrics import is_refusal

import agent_smoke
import answer_tuning
import route_eval
import selfrag_golden as sg

RUNS = {"GraphRAG FINAL": "phase1.json", "agent, no web": "phase3b_tune_noweb.json",
        "agent, with web": "phase3b_tune_web.json"}


def grade(g, answer: str) -> tuple[float, float]:
    if g.question_type == "no_answer":
        x = float(is_refusal(answer))
        return x, x
    if is_refusal(answer):
        return 0.0, 0.0
    return route_eval.grade(g.question, g.ground_truth, answer), answer_tuning.grade3(g, answer)


def main():
    golden = {g.id: g for g in load_golden(sg.golden_name("tune"))}
    before = agent_smoke.snapshot()
    out = {}
    for name, file in RUNS.items():
        rows = json.loads((ab.RESULTS / file).read_text())["questions"]
        per, flips = {}, {}
        for i, r in rows.items():
            cands = (r.get("selfrag") or {}).get("candidates") or []
            a_all = next((c["answer"] for c in cands if c["name"] == "A_all"), None)
            if a_all is None or a_all == r["answer"]:
                per[i] = {"correct": r["correct"], "correct3": r["correct3"], "answer": r["answer"], "same": True}
                continue
            c1, c3 = grade(golden[i], a_all)
            per[i] = {"correct": c1, "correct3": c3, "answer": a_all, "same": False,
                      "selfrag_answer": r["answer"], "selfrag_correct": r["correct"],
                      "picked": r["selfrag"].get("picked")}
            if c1 != r["correct"]:
                flips[i] = f"{r['correct']:g} -> {c1:g}"
        n = len(rows)
        with_sr = (sum(r["correct"] for r in rows.values()) / n, sum(r["correct3"] for r in rows.values()) / n)
        without = (sum(p["correct"] for p in per.values()) / n, sum(p["correct3"] for p in per.values()) / n)
        out[name] = {"with_selfrag": with_sr, "without_selfrag": without, "flips": flips,
                     "differ": [i for i, p in per.items() if not p["same"]], "questions": per}
        print(f"\n{name}: with Self-RAG {with_sr[0]:.3f} / {with_sr[1]:.3f}  ->  without (A_all) "
              f"{without[0]:.3f} / {without[1]:.3f}   (1 vote / 3 votes)")
        print(f"  answers that differ: {len(out[name]['differ'])}; score flips: {flips or 'none'}")
    spent = agent_smoke.spent(before)
    out["spend_usd"] = round(spent, 4)
    (ab.RESULTS / "phase3c_no_selfrag_offline.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nreal spend ${spent:.4f} (grader calls only)")


if __name__ == "__main__":
    main()
