"""Phase 3: the full system with the ladder agent on the multi route, on Self-RAG's tune set (78), without and with
web, graded with the 1-vote and 3-vote graders and compared question by question with GraphRAG FINAL (Phase 1
replay, phase1.json, 0.942).

    router -> none / single (unchanged, replay from the cache)
           -> multi: walk -> enough? -> (walk from another match | IRCoT) -> ... -> web (WEB mode) -> CRAG -> Self-RAG

Only the 39 multi questions can make new calls. Reported per mode: the score, every question that changed against
GraphRAG, how far each multi question climbed the ladder, how often the agent said wrong match / fact missing, how
often the web ran, no_answer questions that broke, and $/question and median time on multi.

Decision rule (proposed to Aditya): the agent without web becomes #8's setup if it scores at least GraphRAG's 0.942
on tune (ties go to the cheaper); the with-web score is reported beside it as what the web adds.

Results: agentic/results/phase3_tune_{noweb,web}.json (first run), phase3b_tune_{noweb,web}.json (amended run:
no known-linked grading for IRCoT/web, no second walk, web on the single route in WEB mode).

Run from agentic_rag_in_depth/ (Qdrant + kg-neo4j running; OpenAI and Serper keys sourced):
    uv run agentic/agent_tune.py
"""

import json
import statistics
from collections import Counter

import agentic_bridge as ab
from eval_harness import load_golden

import agent_loop
import agent_smoke
import agent_web
import answer_tuning
import baseline
import crag
import crag_eval
import route_eval
import self_rag
import selfrag_golden as sg

TAG = "phase3b"  # the amended run: no known-linked grading for IRCoT/web, no second walk, web on single (WEB mode)


def run(golden: dict, split: str = "tune", web: bool = False) -> dict:
    agent_loop.install(web=web, single="always")  # Phase 3b: web on every single question
    picks = {}
    rows = route_eval.run_route("agent_web" if web else "agent", golden, baseline.routed(picks),
                                golden_set=sg.golden_name(split))
    for i, row in rows.items():
        q = golden[i].question
        log = self_rag.LOG.get(q, {})
        row["route"], row["kind"] = picks[q], sg.kind(i)
        row["correct3"] = answer_tuning.grade3(golden[i], row["answer"])
        row["crag"] = crag.LOG.get(q)
        row["selfrag"] = {k: log.get(k) for k in ("sure", "picked", "good", "retries", "candidates")}
        row["agent"] = agent_loop.LOG.get(q) if picks[q] == "multi" else None
        row["single_web"] = agent_loop.SINGLE_LOG.get(q) if picks[q] == "single" else None
    return rows


def path(agent: dict) -> str:
    """'walk', 'walk > ircot', 'walk > walk > ircot > web', ..."""
    return " > ".join({"graph_walk": "walk", "ircot": "ircot", "web_search": "web"}[t] for t in agent["tools"])


def report(rows: dict, base: dict, golden: dict, prev: dict) -> dict:
    s = crag_eval.summary(rows, None)
    multi = [i for i in rows if base[i]["route"] == "multi"]
    single = [i for i in rows if base[i]["route"] == "single"]
    agents = {i: rows[i]["agent"] for i in multi if rows[i]["agent"]}
    s["multi"] = {"n": len(multi), "graphrag": sum(base[i]["correct"] for i in multi),
                  "agent": sum(rows[i]["correct"] for i in multi)}
    s["single"] = {"n": len(single), "graphrag": sum(base[i]["correct"] for i in single),
                   "now": sum(rows[i]["correct"] for i in single)}
    s["changed"] = {i: f"{base[i]['correct']:g} -> {rows[i]['correct']:g}" for i in rows
                    if rows[i]["correct"] != base[i]["correct"]}
    s["vs_phase3"] = {i: f"{prev[i]['correct']:g} -> {rows[i]['correct']:g}" for i in rows
                      if rows[i]["correct"] != prev[i]["correct"]}
    s["route_changed"] = [i for i in rows if rows[i]["route"] != base[i]["route"]]
    s["paths"] = dict(Counter(path(a) for a in agents.values()).most_common())
    s["ends"] = dict(Counter(a["end"] for a in agents.values()))
    s["web_ran"] = [i for i, a in agents.items() if a["web"]]
    s["no_answer_broken"] = [i for i in rows if golden[i].question_type == "no_answer"
                             and rows[i]["correct"] < base[i]["correct"]]
    s["multi_cost"] = {"graphrag": sum(base[i]["cost"] for i in multi) / len(multi),
                       "agent": sum(rows[i]["cost"] for i in multi) / len(multi)}
    s["multi_p50_ms"] = {"graphrag": statistics.median(base[i]["ms"] for i in multi),
                         "agent": statistics.median(rows[i]["ms"] for i in multi)}
    return s


def main():
    golden = {g.id: g for g in load_golden(sg.golden_name("tune"))}
    phase1 = json.loads((ab.RESULTS / "phase1.json").read_text())
    base, b = phase1["questions"], phase1["summary"]
    for web in (False, True):
        mode = "web" if web else "noweb"
        prev = json.loads((ab.RESULTS / f"phase3_tune_{mode}.json").read_text())["questions"]
        before, serper = agent_smoke.snapshot(), dict(agent_web.USAGE)
        print(f"\n=== agent, {'with' if web else 'no'} web: running 78 tune questions ===", flush=True)
        rows = run(golden, web=web)
        s = report(rows, base, golden, prev)
        s["spend"] = {"real_usd": round(agent_smoke.spent(before), 4),
                      "serper_requests": agent_web.USAGE["requests"] - serper["requests"]}
        (ab.RESULTS / f"{TAG}_tune_{mode}.json").write_text(
            json.dumps({"summary": s, "questions": rows}, indent=2, ensure_ascii=False))
        m, sg_ = s["multi"], s["single"]
        print(f"right (78): GraphRAG {b['correct']:.3f} -> agent {s['correct']:.3f} (1 vote); "
              f"{b['correct3']:.3f} -> {s['correct3']:.3f} (3 votes)")
        print(f"multi (39): GraphRAG {m['graphrag']:g} -> agent {m['agent']:g}; "
              f"single ({sg_['n']}): {sg_['graphrag']:g} -> {sg_['now']:g}")
        print(f"changed vs GraphRAG: {s['changed'] or 'none'}; route changed: {s['route_changed'] or 'none'}")
        print(f"changed vs first Phase 3 run: {s['vs_phase3'] or 'none'}")
        print(f"ladder paths: {s['paths']}; ends: {s['ends']}")
        print(f"web ran on multi: {s['web_ran'] or 'none'}; no_answer broken: {s['no_answer_broken'] or 'none'}")
        print(f"multi $/question {s['multi_cost']['graphrag']:.5f} -> {s['multi_cost']['agent']:.5f}; "
              f"multi p50 {s['multi_p50_ms']['graphrag'] / 1000:.1f} -> {s['multi_p50_ms']['agent'] / 1000:.1f} s")
        print(f"real spend ${s['spend']['real_usd']:.4f}; Serper requests {s['spend']['serper_requests']}", flush=True)


if __name__ == "__main__":
    main()
