"""Web rung check, before Phase 3: the smoke runs never reached the web, so this proves the path works.

1. TOOL   agent_tools.web_search on one tune question: Serper answers, each result becomes a passage registered in
          CRAG's chunk table (so the grader, answer step and Self-RAG can read it), the reply is cached on disk, and
          the same query again is a cache hit (no request).
2. FORCED one question end to end in WEB mode with the agent's "enough" overridden to false at every decision, so
          the ladder climbs walk -> IRCoT -> web. Checks: the web ran with the agent's own query, its top 3 lead the
          pool, CRAG graded them, Self-RAG read them, an answer came out. The forced answer is a wiring check only:
          it counts toward no score.

Results: agentic/results/phase3_webcheck.json.

Run from agentic_rag_in_depth/ (Qdrant + kg-neo4j running; OpenAI and Serper keys sourced):
    uv run agentic/agent_webcheck.py
"""

import json

import agentic_bridge as ab
from eval_harness import load_golden

import agent_loop
import agent_smoke
import agent_tools
import agent_web
import baseline
import crag
import ladder
import self_rag
import selfrag_golden as sg

QID = "chain-07"


def ok(name: str, cond: bool, results: dict) -> None:
    results[name] = bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


def main():
    g = {x.id: x for x in load_golden(sg.golden_name("tune"))}[QID]
    agent_loop.install(web=True)
    before = agent_smoke.snapshot()
    checks, out = {}, {"question": QID}

    print(f"1. TOOL: web_search on {QID}")
    o = agent_tools.web_search(g.question)
    requests = agent_web.USAGE["requests"]
    ok("Serper returned results", len(o.hits) > 0, checks)
    ok("every result registered as a chunk", all(h.chunk_id in crag._by_id for h in o.hits), checks)
    ok("CRAG's labeller reads them", all(ladder.labelled(h).startswith("Web › ") for h in o.hits), checks)
    ok("reply cached on disk", agent_web.key(g.question) in json.loads(agent_web.CACHE.read_text()), checks)
    agent_tools.web_search(g.question)
    ok("same query again = cache hit, no request", agent_web.USAGE["requests"] == requests, checks)
    out["tool"] = {"results": len(o.hits), "sources": [h.path[1][:60] for h in o.hits]}
    print(f"  {len(o.hits)} results; Serper requests so far {agent_web.USAGE['requests']}")

    print(f"\n2. FORCED: {QID} end to end, agent's 'enough' overridden to false")
    real, said = agent_loop.decide, []

    def never_enough(question, obs, stage):
        d, reply = real(question, obs, stage)
        said.append({"stage": stage, "agent_said_enough": d.get("enough"), "web_query": d.get("web_query")})
        d["enough"] = False
        return d, reply

    agent_loop.decide = never_enough
    try:
        picks = {}
        answer = baseline.routed(picks)(g.question).answer
    finally:
        agent_loop.decide = real
    log, clog = agent_loop.LOG[g.question], crag.LOG[g.question]
    slog = self_rag.LOG.get(g.question, {})
    web_step = next(s for s in log["steps"] if s.get("tool") == "web_search")
    web_ids = set(web_step["hits"][:agent_loop.WEB_TOP])
    scores = dict(zip(log["pool"], clog["scores"]))
    ok("routed to multi", picks[g.question] == "multi", checks)
    ok("ladder climbed walk -> ircot -> web", log["tools"] == ["graph_walk", "ircot", "web_search"], checks)
    ok("web's top 3 lead the pool", set(log["pool"][:agent_loop.WEB_TOP]) == web_ids, checks)
    ok("CRAG graded the web passages", all(i in scores for i in web_ids), checks)
    ok("Self-RAG wrote candidates over the pool", len(slog.get("candidates") or []) > 1, checks)
    ok("an answer came out", bool(answer.strip()), checks)
    out["forced"] = {"decisions": said, "web_query": web_step["args"].get("query") or g.question,
                     "pool": log["pool"], "crag_action": clog["action"],
                     "web_scores": {i: scores.get(i) for i in web_step["hits"][:agent_loop.WEB_TOP]},
                     "kept": clog["kept"], "picked": slog.get("picked"), "answer": answer}
    print(f"  web query: {out['forced']['web_query']!r}")
    print(f"  pool: {len(log['pool'])} passages, first 3 = web; CRAG {clog['action']}; "
          f"web passage scores {list(out['forced']['web_scores'].values())}")
    print(f"  Self-RAG picked {slog.get('picked')}; answer: {answer[:160]}")

    real_usd = agent_smoke.spent(before)
    out |= {"checks": checks, "spend": {"real_usd": round(real_usd, 4), "serper_requests": agent_web.USAGE["requests"],
                                        "serper_cached": agent_web.USAGE["cached"]}}
    (ab.RESULTS / "phase3_webcheck.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n{sum(checks.values())}/{len(checks)} checks pass; real spend ${real_usd:.4f}; "
          f"Serper requests {agent_web.USAGE['requests']} (cache hits {agent_web.USAGE['cached']})")


if __name__ == "__main__":
    main()
