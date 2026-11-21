"""Phase 1: the base the agent plugs into. GraphRAG #7 FINAL (graph walk + linked grading + IRCoT fallback) on
Self-RAG's tune set (78), run live through the bridge:

    few-shot router -> none (+ sure check) / single (f6a) / multi (graph walk -> CRAG linked -> IRCoT if INCORRECT)
                    -> CRAG 3f -> Self-RAG layer -> answer

and graded with the 1-vote and 3-vote graders. Every LLM call should come from the cache; run it with a dummy OpenAI
key and any uncached call fails instead of spending. It must match GraphRAG's saved run
(phase4_tune_linked_fallback.json, 0.942) question by question: score, route, CRAG action, Self-RAG pick and answer
text, or the reuse is broken.

Results go to agentic/results/phase1.json.

Run from agentic_rag_in_depth/ (Qdrant + kg-neo4j running):
    OPENAI_API_KEY=sk-dummy uv run agentic/replay_graph.py
"""

import json

import agentic_bridge as ab
from eval_harness import load_golden

import crag_eval
import graph_tune
import selfrag_golden as sg

RESULTS = ab.RESULTS / "phase1.json"
SAVED = ab.GRAG_RESULTS / "phase4_tune_linked_fallback.json"
FIELDS = ("correct", "correct3", "route", "answer")


def main():
    golden = {g.id: g for g in load_golden(sg.golden_name("tune"))}
    rows = graph_tune.run(golden, "tune", **ab.FINAL)
    saved = json.loads(SAVED.read_text())
    now = crag_eval.summary(rows, None)
    then = saved["summary"]

    action = lambda row: row["crag"]["action"] if row["crag"] else None
    picked = lambda row: (row["selfrag"] or {}).get("picked")
    differs = {}
    for i in golden:
        old = saved["questions"][i]
        d = [f for f in FIELDS if rows[i][f] != old[f]]
        d += ["action"] * (action(rows[i]) != action(old)) + ["picked"] * (picked(rows[i]) != picked(old))
        if d:
            differs[i] = d
    RESULTS.write_text(json.dumps({"summary": now, "saved_summary": then, "differs": differs, "questions": rows},
                                  indent=2, ensure_ascii=False))

    print("\n| | now | GraphRAG saved |\n|---|---|---|")
    for key in ("correct", "correct3"):
        print(f"| {key} | {now[key]:.3f} | {then[key]:.3f} |")
    print(f"| routes | {now['routes']} | {then['routes']} |")
    print(f"| $/question (original calls) | {now['cost_per_question']:.5f} | {then['cost_per_question']:.5f} |")
    print(f"| p50 s | {now['p50_ms'] / 1000:.1f} | {then['p50_ms'] / 1000:.1f} |")
    print(f"\nquestions that differ: {differs or 'none'}")


if __name__ == "__main__":
    main()
