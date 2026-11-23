"""The agent on the router's multi route: a fixed escalation ladder where gpt-4o-mini decides only "enough?"
(Aditya's ladder; ReAct's think -> act -> observe, Yao et al. 2022). In WEB mode the single route also searches
the web every time.

MULTI
    step 1  graph_walk from the closest match                 fixed (= GraphRAG's multi search, cached)
            enough? yes -> answer
    step 2  ircot                                             forced
            enough? yes -> answer                             (asked only in WEB mode: without web nothing follows)
    step 3  web_search with the agent's query                 forced, WEB mode only; answers with what it has

    -> POOL     the web's top 3 first (when it ran), then the walk's passages, then IRCoT's; no repeats, <= 15,
                every step keeps 3 seats
    -> PREPARE  graph passages are graded with the record line that brought them (GraphRAG's linked grading)
    -> ANSWER   CRAG 3f (the grading) -> Self-RAG -> answer (routes.answer_from, unchanged)

SINGLE (WEB mode only)
    f6a top 5 + web top 3 (the question as the query) -> CRAG -> Self-RAG -> answer. The library passages come first:
    here the web is added every time, not reached because the library fell short.

Phase 3 amendments (Aditya): (1) IRCoT and web passages are graded against the plain question. The agent's own
"known" facts were wrong three times (chain-06 smoke v2, twin-03, gap-02 on tune) and a wrong fact made the grader
accept wrong passages; the graph's link comes from a record line, so it stays. (2) No second walk: on tune the agent
said "wrong match" twice and both times named the match already walked from; IRCoT already recovers from a wrong
seed (chain-06). The agent never picks tools: two smoke runs with free choice gave two strategies, both worse.

Every decision goes through llm_cache (JSON mode), so a replay of the same steps is free.
install(web) puts agent_multi (and in WEB mode single_web) in the router's table over GraphRAG FINAL.
"""

import json

import agentic_bridge as ab
import agent_tools
import baseline
import crag
import graph_route
import ladder
import routes
import stacked
from llm_cache import chat

PREVIEW = 400  # characters of each passage the agent reads
POOL_MAX = 15  # IRCoT's own pool can reach 15 (3 searches x 5)
WEB_TOP = 3  # web results that enter the pool
MIN_SEATS = 3  # places every step keeps before the free places are shared out
WEB = {"on": False}
LOG: dict[str, dict] = {}
SINGLE_LOG: dict[str, dict] = {}

PROMPT = """You check whether the passages found so far answer a cricket question.

Question: {question}

Steps so far:
{steps}

Check that each passage is about the exact match, ground or player the question means: similar matches are easy to \
confuse. "enough" is true only if the passages answer every part of the question.
{task}
Reply as JSON: {fmt}"""
TASK = {"plain": "",
        "web": "If they do not answer it, the next step searches Google: write a short web search for what is missing."}
FMT = {"plain": '{"enough": true or false, "missing": "<what is still missing, or empty>"}',
       "web": '{"enough": true or false, "missing": "<what is still missing, or empty>", '
              '"web_query": "<short search, or empty>"}'}


def render(obs: list[agent_tools.Obs]) -> str:
    """Each step: the call, the tool's note, then a preview of each passage not shown before (numbered over all)."""
    out, shown = [], set()
    for step, o in enumerate(obs, start=1):
        lines = [f"{step}. {o.tool} {json.dumps(o.args, ensure_ascii=False)}", o.text]
        for h in o.hits:
            if h.chunk_id in shown:
                continue
            shown.add(h.chunk_id)
            body = " ".join(h.text.split())[:PREVIEW]
            lines.append(f"[{len(shown)}] {' › '.join(h.path)}: {body}")
        out.append("\n".join(lines))
    return "\n\n".join(out)


def next_rung(obs: list) -> str | None:
    """What runs if the agent says "not enough": ircot, then the web in WEB mode; None when the ladder is used up."""
    tools = [o.tool for o in obs]
    if "ircot" not in tools:
        return "ircot"
    if WEB["on"] and "web_search" not in tools:
        return "web_search"
    return None


def prompt(question: str, obs: list, stage: str) -> str:
    return PROMPT.format(question=question, steps=render(obs), task=TASK[stage], fmt=FMT[stage])


def decide(question: str, obs: list, stage: str) -> tuple[dict, object]:
    reply = chat(prompt(question, obs, stage), json_mode=True)
    try:
        d = json.loads(reply.text)
    except json.JSONDecodeError:
        d = {}
    return (d if isinstance(d, dict) else {}), reply


def build_pool(obs: list) -> list:
    """The most targeted evidence first. Web, when it ran, is the top WEB_TOP: the ladder reaches it only after the
    library fell short, and its query names the missing fact. Then the walk's passages (GraphRAG: read after
    look-alikes they broke answers), then IRCoT's. No repeats, at most POOL_MAX.

    Every step keeps seats: each first gets its best MIN_SEATS, then the free places go in the same order. So the
    cut trims the end of a long list (IRCoT's), never a whole step. With the walk and IRCoT only, the pool is the
    walk's <= 9 then IRCoT's best, as before."""
    groups = ([o.hits[:WEB_TOP] for o in obs if o.tool == "web_search"]
              + [o.hits for o in reversed(obs) if o.tool == "graph_walk"]
              + [o.hits for o in obs if o.tool in ("ircot", "f6a_search")])  # f6a_search: the single route's step
    seen, deduped = set(), []
    for hits in groups:  # a passage belongs to the first step that found it
        mine = [h for h in hits if h.chunk_id not in seen]
        seen |= {h.chunk_id for h in mine}
        deduped.append(mine)
    take = [min(MIN_SEATS, len(g)) for g in deduped]
    free = POOL_MAX - sum(take)
    for n, g in enumerate(deduped):  # free places, in priority order
        extra = max(0, min(len(g) - take[n], free))
        take[n] += extra
        free -= extra
    return [h for g, n in zip(deduped, take) for h in g[:n]][:POOL_MAX]


def link(pool: list, obs: list) -> None:
    """PREPARE for CRAG: graph passages are graded with the record line that brought them (graph_route._HOP_Q, read
    by CRAG's grader); every other passage with the plain question."""
    for o in obs:
        graph_route._HOP_Q.update({ladder.labelled(h): o.linked[h.chunk_id] for h in pool if h.chunk_id in o.linked})


def agent_multi(question: str):
    routes._start()
    first = agent_tools.graph_walk(question)  # step 1, fixed
    obs, replies, end = [first], [], "ladder_end"
    steps = [{"decision": {}, "fixed": True, "tool": "graph_walk", "args": first.args, "note": first.text,
              "hits": [h.chunk_id for h in first.hits]}]
    while (rung := next_rung(obs)) is not None:  # without web the ladder ends after ircot, with no question asked
        d, reply = decide(question, obs, "web" if rung == "web_search" else "plain")
        replies.append(reply)
        step = {"decision": {k: d.get(k) for k in ("enough", "missing", "web_query") if k in d}}
        steps.append(step)
        if d.get("enough") is True:
            end = "enough"
            break
        args = {"query": str(d.get("web_query") or d.get("missing") or "").strip()} if rung == "web_search" else {}
        o = agent_tools.run(rung, args, question)
        obs.append(o)
        step |= {"tool": o.tool, "args": o.args, "note": o.text, "hits": [h.chunk_id for h in o.hits], "log": o.log}

    pool = build_pool(obs)
    if not pool:  # no step found a passage: GraphRAG's no-path rule, plain f6a
        pool = routes.f6a(question)
    link(pool, obs)
    tool_replies = [r for o in obs for r in o.replies]
    routes.USAGE.update(searches=len(obs), thoughts=[json.dumps(s["decision"]) for s in steps],
                        added=[[h.chunk_id for h in o.hits] for o in obs])
    LOG[question] = {"steps": steps, "tools": [o.tool for o in obs], "end": end, "cap_hit": end == "ladder_end",
                     "web": any(o.tool == "web_search" for o in obs), "pool": [h.chunk_id for h in pool],
                     "agent_calls": len(replies)}
    try:
        return routes.answer_from(question, pool, replies + tool_replies)
    finally:
        graph_route._HOP_Q.clear()


def single_ladder(question: str):
    """WEB mode's single route (run D on): the multi ladder's rule on one library step.

        f6a top 5 -> AGENT: enough?  yes -> answer from f6a's passages (the same pool as without web)
                                     no  -> web search with the agent's query -> web top 3 first, then f6a's 5
    """
    routes._start()
    library = agent_tools.Obs("f6a_search", {}, routes.f6a(question), "Library search for the question.")
    obs = [library]
    d, reply = decide(question, obs, "web")
    step = {"decision": {k: d.get(k) for k in ("enough", "missing", "web_query") if k in d}}
    if d.get("enough") is not True:
        o = agent_tools.web_search(question, str(d.get("web_query") or d.get("missing") or "").strip())
        obs.append(o)
        step |= {"tool": "web_search", "args": o.args, "hits": [h.chunk_id for h in o.hits]}
    pool = build_pool(obs)
    routes.USAGE.update(searches=len(obs), thoughts=[json.dumps(step["decision"])],
                        added=[[h.chunk_id for h in o.hits] for o in obs])
    SINGLE_LOG[question] = {"library": [h.chunk_id for h in library.hits], "step": step,
                            "web": len(obs) > 1, "pool": [h.chunk_id for h in pool]}
    return routes.answer_from(question, pool, [reply])


def single_web(question: str):
    """WEB mode's single route in Phase 3b: f6a's top 5, then the web's top 3 for the question, every time."""
    routes._start()
    library = routes.f6a(question)
    web = agent_tools.web_search(question).hits[:WEB_TOP]
    seen = {h.chunk_id for h in library}
    pool = library + [h for h in web if h.chunk_id not in seen]
    routes.USAGE.update(searches=2, thoughts=[], added=[[h.chunk_id for h in library], [h.chunk_id for h in web]])
    SINGLE_LOG[question] = {"library": [h.chunk_id for h in library], "web": [h.chunk_id for h in web]}
    return routes.answer_from(question, pool, [])


def install(web: bool = False, selfrag: bool = True, single: str = "ladder") -> None:
    """GraphRAG FINAL (router, CRAG 3f + linked grading, Self-RAG), with the agent on the multi route; in WEB mode
    the single route adds the web. The none route's "unsure -> single" keeps the plain single route.

    selfrag=False removes Self-RAG's candidates, critic and retry (offline check, phase3c: with gpt-4o-mini as the
    critic they changed no score without web and rejected right web answers with it): every route answers with
    CRAG's own answer. Self-RAG's "are you sure?" step on the none route stays.

    single (WEB mode only): "ladder" = f6a -> enough? -> web (single_ladder, run D on); "always" = f6a + web on every
    question (single_web, Phase 3b)."""
    graph_route.install(ab.SELFRAG_CONFIG, **ab.FINAL)
    if not selfrag:
        routes.answer_from = crag.crag_answer_from  # sure_or_search stays on the none route
    WEB["on"] = web
    LOG.clear()
    SINGLE_LOG.clear()
    baseline.ROUTE_FNS["multi"] = agent_multi
    baseline.ROUTE_FNS["single"] = ({"ladder": single_ladder, "always": single_web}[single] if web
                                    else stacked.F6A_ROUTES["single"])
