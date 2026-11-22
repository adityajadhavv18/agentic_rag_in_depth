"""Phase 2 dry check, free (no new LLM calls; run it with a dummy key and any uncached call fails).

1. CUES  For the 39 tune questions the router sends to multi: the 3 closest match names for the question, the gap between
         the first two scores, and what GraphRAG FINAL did with it (Phase 1 replay: its seed, CRAG's action, right or
         wrong). Does a small gap (a near-tie seed) go with the walk's mistakes?
2. WIRING   both library tools once on one question, through the cache: each must return passages and a note.
3. PROMPT   the agent's first decision prompt (after the fixed walk) and its length, to price a step.

Results: agentic/results/phase2_dry.json.

Run from agentic_rag_in_depth/ (Qdrant + kg-neo4j running):
    OPENAI_API_KEY=sk-dummy uv run agentic/agent_dry.py
"""

import json

import agentic_bridge as ab
from eval_harness import load_golden

import agent_loop
import agent_tools
import selfrag_golden as sg

NEAR = 0.005  # GraphRAG's near-tie line (9 of 39 multi questions)
WIRE_ID = "chain-07"


def main():
    golden = {g.id: g for g in load_golden(sg.golden_name("tune"))}
    base = json.loads((ab.RESULTS / "phase1.json").read_text())["questions"]
    multi = [i for i in golden if base[i]["route"] == "multi"]
    agent_loop.install(web=False)

    cues = {}
    for i in multi:
        top = agent_tools.top_matches(golden[i].question)
        walk = base[i]["walk"] or {}
        cues[i] = {"correct": base[i]["correct"], "gap": round(top[0][2] - top[1][2], 3), "top": top,
                   "seed_used": walk.get("via_name"), "action": (base[i]["crag"] or {}).get("action"),
                   "ircot_fallback": bool(walk.get("ircot_fallback"))}
    print("\n1. CUES (sorted by gap; * = GraphRAG wrong or half)")
    for i, c in sorted(cues.items(), key=lambda kv: kv[1]["gap"]):
        mark = "*" if c["correct"] < 1 else " "
        print(f"{mark} {i:10s} gap {c['gap']:.3f}  {c['top'][0][2]:.3f} {c['top'][0][1][:55]!r:58s} "
              f"vs {c['top'][1][1][:40]!r}  CRAG {c['action']}")
    near = [i for i, c in cues.items() if c["gap"] < NEAR]
    wrong = [i for i, c in cues.items() if c["correct"] < 1]
    print(f"\nnear-ties (gap < {NEAR}): {len(near)}; of them GraphRAG wrong: {[i for i in near if i in wrong]}")
    print(f"GraphRAG wrong on multi: {wrong}")

    print(f"\n2. WIRING on {WIRE_ID} (library tools; web is checked apart)")
    q = golden[WIRE_ID].question
    wiring, obs = {}, []
    for tool, args in (("graph_walk", {"match": ""}), ("ircot", {})):
        o = agent_tools.run(tool, args, q)
        obs.append(o)
        wiring[tool] = {"hits": [h.chunk_id for h in o.hits], "note": o.text}
        print(f"  {tool:12s} {len(o.hits):2d} passages | {o.text[:110]!r}")

    print("\n3. PROMPT (the agent's first decision, after the fixed walk)")
    prompt = agent_loop.prompt(q, obs[:1], "first")
    print(f"  ~{len(prompt) // 4} tokens\n\n{prompt}")

    (ab.RESULTS / "phase2_dry.json").write_text(json.dumps(
        {"near": near, "wrong": wrong, "cues": cues, "wiring": wiring, "prompt_tokens": len(prompt) // 4},
        indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
