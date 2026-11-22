"""Phase 2 smoke: the full system with the agent on the multi route, on 5 tune questions, without and with web,
against GraphRAG FINAL's answers (Phase 1 replay, phase1.json):

    chain-06   GraphRAG 0.5: near-tie seed (1st vs 2nd Test of the 2024-25 series), answer added a wrong venue
    twin-03    GraphRAG needed its IRCoT fallback (the walk's passages were all rejected)
    alike-06   look-alike finals (2022 vs 2021 Men's T20 World Cup, gap 0.003): the agent must not switch seeds
    chain-07   right: player of the match -> school; needs linked grading
    pair-01    right: a pair question

Spend is measured, not estimated: the new entries in every LLM cache (this build's llm.json, Contextual's rerank and
HyDE caches, Fusion's rewrite caches) priced at gpt-4o-mini rates. Web: Serper requests vs cache hits.

Results: agentic/results/phase2_smoke.json.

Run from agentic_rag_in_depth/ (Qdrant + kg-neo4j running; OpenAI and Serper keys sourced):
    uv run agentic/agent_smoke.py
"""

import json
import time

import agentic_bridge as ab
from eval_harness import load_golden, settings
from eval_harness.metrics import cost_usd, is_refusal

import agent_loop
import agent_web
import baseline
import route_eval
import self_rag
import selfrag_golden as sg

IDS = ("chain-06", "twin-03", "alike-06", "chain-07", "pair-01")
CACHES = [ab.ROOT / "data" / "cache" / "llm.json",
          *(ab.RAGS / "contextual_rag_in_depth" / "data" / "cricket" / "cache").glob("*.json"),
          *(ab.RAGS / "rag_fusion_in_depth" / "data" / "cache").glob("*.json")]


def snapshot() -> dict:
    return {p: set(json.loads(p.read_text())) if p.exists() else set() for p in CACHES}


def spent(before: dict) -> float:
    """Real dollars: the calls that were not in any cache before the run."""
    total = 0.0
    for p, keys in before.items():
        store = json.loads(p.read_text()) if p.exists() else {}
        for k in set(store) - keys:
            e = store[k]
            total += cost_usd(e.get("tokens_in", 0), e.get("tokens_out", 0), settings.llm_model) or 0.0
    return total


def grade(g, answer: str) -> float:
    if g.question_type == "no_answer":
        return float(is_refusal(answer))
    return 0.0 if is_refusal(answer) else route_eval.grade(g.question, g.ground_truth, answer)


def trace(log: dict) -> str:
    parts = []
    for s in log["steps"]:
        d = s["decision"]
        if "tool" in s:
            arg = next((v for v in (s["args"] or {}).values() if v), "")
            parts.append(f"{s['tool']}({arg[:45]})" + " [fixed]" * bool(s.get("fixed")))
        else:
            parts.append("enough" if d.get("enough") else f"stop({d.get('tool')})")
    return " -> ".join(parts) + (f" [{log['end']}]" if log.get("cap_hit") else "")


def main():
    golden = {g.id: g for g in load_golden(sg.golden_name("tune"))}
    base = json.loads((ab.RESULTS / "phase1.json").read_text())["questions"]
    before = snapshot()
    out = {}
    for web in (False, True):
        mode = "web" if web else "no_web"
        agent_loop.install(web=web)
        picks, rows = {}, {}
        pipeline = baseline.routed(picks)
        print(f"\n=== agent, {mode.replace('_', ' ')} ===")
        for i in IDS:
            g = golden[i]
            start = time.perf_counter()
            o = pipeline(g.question)
            secs = time.perf_counter() - start
            log = agent_loop.LOG.get(g.question, {"steps": [], "tools": [], "cap_hit": False, "web": False})
            rows[i] = {"graphrag": base[i]["correct"], "now": grade(g, o.answer), "route": picks[g.question],
                       "trace": trace(log), "answer": o.answer, "graphrag_answer": base[i]["answer"],
                       "cost": cost_usd(o.tokens_in, o.tokens_out, settings.llm_model), "secs": round(secs, 1),
                       "agent": log, "selfrag_picked": self_rag.LOG.get(g.question, {}).get("picked")}
            r = rows[i]
            print(f"\n{i}: GraphRAG {r['graphrag']:g} -> agent {r['now']:g}   {r['trace']}   "
                  f"${r['cost']:.4f}  {secs:.1f} s")
            print(f"  agent:    {o.answer[:220]}")
            print(f"  GraphRAG: {base[i]['answer'][:220]}")
        out[mode] = rows
        print(f"\nright ({mode}): GraphRAG {sum(r['graphrag'] for r in rows.values()):g}/5 -> "
              f"agent {sum(r['now'] for r in rows.values()):g}/5")
    real = spent(before)
    out["spend"] = {"real_usd": round(real, 4), "serper_requests": agent_web.USAGE["requests"],
                    "serper_cached": agent_web.USAGE["cached"]}
    (ab.RESULTS / "phase2_smoke.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nreal spend ${real:.4f}; Serper requests {agent_web.USAGE['requests']} "
          f"(cache hits {agent_web.USAGE['cached']})")


if __name__ == "__main__":
    main()
