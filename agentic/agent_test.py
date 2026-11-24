"""Phase 4, the one test run: 20 fresh questions (golden_cricket_agentic_test_v1.jsonl, agent_golden.py) that no
system has seen, through three setups, in this order:

    graphrag   GraphRAG #7 FINAL as it shipped: graph walk + linked grading + IRCoT fallback, Self-RAG on,
               the old library (no record pages), no web
    agent_c    #8 FINAL (run C on tune, 0.974): the ladder agent on multi, record pages on, Self-RAG off, no web
    agent_d    C + web as a last resort on both routes (run D on tune, 0.962)

GraphRAG runs first because the agent's install swaps routes.answer_from (Self-RAG off) and the record pages'
collection; it starts from the state every earlier build ran in.

The 3 record questions (alike-31..33) have their answer only in the record pages, which only #8 has, so the main
comparison is on the other 17 and the record questions are reported apart.

It saves agentic/results/test_once.json and refuses to run again once that file exists.

Run from agentic_rag_in_depth/ (Qdrant + kg-neo4j running; OpenAI and Serper keys sourced), once:
    uv run python -I -c "import sys; sys.path.insert(0,'agentic'); import agent_test; agent_test.main()"
"""

import json
import sys

import agentic_bridge as ab
from eval_harness import load_golden

import agent_loop
import agent_smoke
import agent_tune
import agent_web
import answer_tuning
import baseline
import corpus_gap
import crag
import crag_eval
import graph_route
import graph_tune
import route_eval
import self_rag

NAME = "golden_cricket_agentic_test_v1.jsonl"
RESULT = ab.RESULTS / "test_once.json"
KINDS = ("chain", "pair", "twin", "alike", "solo", "gap", "sure", "none")
RECORD_QS = ("alike-31", "alike-32", "alike-33")


def kind(gid: str) -> str:
    return next(k for k in KINDS if gid.startswith(k + "-"))


def agent_rows(name: str, golden: dict, web: bool) -> dict:
    """agent_loop + record pages, Self-RAG off; the same fields corpus_gap.check keeps."""
    agent_loop.install(web=web, selfrag=False)
    corpus_gap.install(True)
    record_docs = {e["doc_id"] for e in corpus_gap.manifest()}
    picks = {}
    rows = route_eval.run_route(name, golden, baseline.routed(picks), golden_set=NAME)
    for i, row in rows.items():
        q = golden[i].question
        row["route"], row["kind"] = picks[q], kind(i)
        row["correct3"] = answer_tuning.grade3(golden[i], row["answer"])
        row["crag"] = crag.LOG.get(q)
        row["selfrag"] = {k: self_rag.LOG.get(q, {}).get(k) for k in ("sure", "picked")}
        row["agent"] = agent_loop.LOG.get(q) if picks[q] == "multi" else None
        row["single_web"] = agent_loop.SINGLE_LOG.get(q) if picks[q] == "single" else None
        row["record_passages"] = [c for c in (row["crag"] or {}).get("kept", []) if c.split("::")[0] in record_docs]
    return rows


def score(rows: dict, ids, key: str = "correct") -> float:
    return sum(rows[i][key] for i in ids)


def summarise(rows: dict, base: dict, golden: dict, prev: dict, spend: float) -> dict:
    s = crag_eval.summary(rows, None)
    main17 = [i for i in rows if i not in RECORD_QS]
    s["main17"] = score(rows, main17) / len(main17)
    s["main17_3votes"] = score(rows, main17, "correct3") / len(main17)
    s["records3"] = f"{score(rows, RECORD_QS):g}/3"
    s["by_kind"] = {k: f"{score(rows, [i for i in rows if kind(i) == k]):g}/{sum(kind(i) == k for i in rows)}"
                    for k in KINDS}
    multi = [i for i in rows if rows[i]["route"] == "multi"]
    s["routes"] = {r: sum(rows[i]["route"] == r for i in rows) for r in ("none", "single", "multi")}
    s["multi"] = f"{score(rows, multi):g}/{len(multi)}"
    s["spend_usd"] = round(spend, 4)
    if base is not None:
        s["changed_vs_graphrag"] = {i: f"{base[i]['correct']:g} -> {rows[i]['correct']:g}" for i in rows
                                    if rows[i]["correct"] != base[i]["correct"]}
        s["route_changed"] = [i for i in rows if rows[i]["route"] != base[i]["route"]]
    if prev is not None:
        s["changed_vs_c"] = {i: f"{prev[i]['correct']:g} -> {rows[i]['correct']:g}" for i in rows
                             if rows[i]["correct"] != prev[i]["correct"]}
    agents = {i: rows[i].get("agent") for i in multi if rows[i].get("agent")}
    if agents:
        s["paths"] = {i: agent_tune.path(a) for i, a in agents.items()}
        s["web_ran_multi"] = [i for i, a in agents.items() if a["web"]]
        s["web_ran_single"] = [i for i, r in rows.items() if (r.get("single_web") or {}).get("web")]
        s["used_records"] = [i for i, r in rows.items() if r.get("record_passages")]
    s["no_answer"] = {i: rows[i]["correct"] for i in rows if golden[i].question_type == "no_answer"}
    return s


def show(label: str, s: dict) -> None:
    print(f"\n=== {label}")
    print(f"right (20): {s['correct']:.3f} (1 vote), {s['correct3']:.3f} (3 votes); main 17: {s['main17']:.3f} "
          f"({s['main17_3votes']:.3f}); record questions: {s['records3']}")
    print(f"routes {s['routes']}; multi {s['multi']}; by kind {s['by_kind']}")
    for key in ("changed_vs_graphrag", "changed_vs_c", "route_changed", "paths", "web_ran_single",
                "web_ran_multi", "used_records", "no_answer"):
        if key in s:
            print(f"{key}: {s[key] or 'none'}")
    print(f"$/question {s['cost_per_question']:.5f}; p50 {s['p50_ms'] / 1000:.1f} s; real spend ${s['spend_usd']:.4f}",
          flush=True)


def main():
    if RESULT.exists():
        sys.exit(f"The test has already been run: {RESULT}")
    golden = {g.id: g for g in load_golden(NAME)}
    out = {"file": NAME, "n": len(golden)}

    before = agent_smoke.snapshot()
    print("=== running GraphRAG #7 FINAL on 20 test questions", flush=True)
    graph_tune.sg.golden_name = lambda split: NAME  # graph_tune names Self-RAG's (spent) test file otherwise
    graph = graph_tune.run(golden, split="test", **ab.FINAL)
    for i, row in graph.items():
        row["kind"] = kind(i)
    g = summarise(graph, None, golden, None, agent_smoke.spent(before))
    out["graphrag"] = {"summary": g, "questions": graph}
    show("GraphRAG #7 FINAL (old library, Self-RAG on, no web)", g)

    before = agent_smoke.snapshot()
    print("\n=== running agent C on 20 test questions", flush=True)
    c_rows = agent_rows("agent_test_C", golden, web=False)
    c = summarise(c_rows, graph, golden, None, agent_smoke.spent(before))
    out["agent_c"] = {"summary": c, "questions": c_rows}
    show("Agent C, #8 FINAL (record pages on, Self-RAG off, no web)", c)

    before, serper = agent_smoke.snapshot(), agent_web.USAGE["requests"]
    print("\n=== running agent D on 20 test questions", flush=True)
    d_rows = agent_rows("agent_test_D", golden, web=True)
    d = summarise(d_rows, graph, golden, c_rows, agent_smoke.spent(before))
    d["serper_requests"] = agent_web.USAGE["requests"] - serper
    out["agent_d"] = {"summary": d, "questions": d_rows}
    show("Agent D (C + web as a last resort)", d)
    print(f"Serper requests {d['serper_requests']}")

    RESULT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nsaved {RESULT}")
