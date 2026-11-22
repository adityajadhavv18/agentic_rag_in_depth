"""The agent on the router's multi route: a fixed escalation ladder, with gpt-4o-mini deciding only "enough?" and,
once, "wrong match or missing fact?" (Aditya's ladder; ReAct's think -> act -> observe, Yao et al. 2022).

    step 1  graph_walk from the closest match                       fixed (= GraphRAG's multi search, cached)
            enough? yes -> answer
            no, WRONG MATCH  -> step 2 graph_walk from the match the agent names (from step 1's closest names)
                                step 3 ircot                        forced
                                step 4 web_search                   forced, WEB mode only
            no, FACT MISSING -> step 2 ircot                        forced from here
                                step 3 web_search                   forced, WEB mode only
    after every step but the last: enough? yes -> answer
    the last rung (ircot without web, web with it) answers with everything collected, no question asked

    -> POOL     graph passages first (the latest walk first), then IRCoT's, then the web's, no repeats, <= 15
    -> PREPARE  link(): each passage gets the question CRAG's grader reads: graph passages the link that brought
                them (GraphRAG's linked grading), IRCoT and web passages the facts the agent says are known
    -> ANSWER   CRAG 3f (the grading) -> Self-RAG -> answer (routes.answer_from, unchanged)

The agent never picks tools freely: two smoke runs gave two different tool strategies, both worse than GraphRAG.
Its "not enough" is the reliable signal and is what moves a question up the ladder. A second walk only runs when the
first one started from the wrong match: when the match was right, a walk from another match only adds look-alikes.
If the agent says "wrong match" but names a match already walked from, or none, the ladder goes to ircot.

Every decision goes through llm_cache (JSON mode), so a replay of the same steps is free. Web is always the last
rung, after the library searches (Aditya's "sequential" choice): the two modes differ only there.

install(web) puts agent_multi in the router's table over GraphRAG FINAL (linked grading on).
"""

import json

import agentic_bridge as ab
import agent_tools
import baseline
import graph_route
import ladder
import routes
from llm_cache import chat

PREVIEW = 400  # characters of each passage the agent reads
POOL_MAX = 15  # IRCoT's own pool can reach 15 (3 searches x 5)
POOL_ORDER = {"graph_walk": 0, "ircot": 1, "web_search": 2}
WEB = {"on": False}
LOG: dict[str, dict] = {}

PROMPT = """You check whether the passages found so far answer a cricket question. Step 1 was a graph walk: it \
started from the match name closest to the question and followed the match's record (venue, player of the match, \
top batter or bowler) to an article.

Question: {question}

Steps so far:
{steps}

Check that each passage is about the exact match, ground or player the question means: similar matches are easy to \
confuse. "enough" is true only if the passages answer every part of the question.
{task}
Reply as JSON: {fmt}"""
TASK = {
    "first": """If they do not answer it, say why:
- "wrong_match": the walk started from a different match than the question means. Name the right match, copied \
exactly from the closest match names in step 1.
- "fact_missing": the match is right, but a fact the question asks for is not in the passages.""",
    "web": "If they do not answer it, the next step searches Google: write a short web search for what is missing.",
    "plain": "",
}
FMT = {
    "first": '{"known": "<facts the passages establish, such as which match, ground or player the question means; '
             'or empty>", "enough": true or false, "problem": "wrong_match" or "fact_missing" or "", '
             '"match": "<exact match name, or empty>"}',
    "web": '{"known": "<facts the passages establish; or empty>", "enough": true or false, '
           '"missing": "<what is still missing, or empty>", "web_query": "<short search, or empty>"}',
    "plain": '{"known": "<facts the passages establish; or empty>", "enough": true or false, '
             '"missing": "<what is still missing, or empty>"}',
}


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
    """What runs if the agent says "not enough": "choice" after step 1 (walk again or ircot), then ircot, then the
    web in WEB mode; None when the ladder is used up (then there is nothing to ask)."""
    tools = [o.tool for o in obs]
    if len(obs) == 1:
        return "choice"
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
    """Graph passages first (GraphRAG: read after look-alikes they broke answers), the latest walk first (a second
    walk means the first started from the wrong match), then IRCoT's, then the web's; no repeats."""
    ranked = sorted(enumerate(obs), key=lambda io: (POOL_ORDER[io[1].tool],
                                                    -io[0] if io[1].tool == "graph_walk" else io[0]))
    pool, seen = [], set()
    for _, o in ranked:
        for h in o.hits:
            if h.chunk_id not in seen:
                seen.add(h.chunk_id)
                pool.append(h)
    return pool[:POOL_MAX]


def link(question: str, pool: list, obs: list, known: str) -> None:
    """PREPARE for CRAG: the question its grader reads for each passage. Graph passages: their link; the rest: the
    known facts. No LLM call here; CRAG's grader reads graph_route._HOP_Q."""
    for o in obs:
        graph_route._HOP_Q.update({ladder.labelled(h): o.linked[h.chunk_id] for h in pool if h.chunk_id in o.linked})
    if known:
        with_known = f"{question} (Known so far: {known.rstrip('.')}.)"
        for h in pool:
            graph_route._HOP_Q.setdefault(ladder.labelled(h), with_known)


def second_step(d: dict, obs: list) -> tuple[str, dict]:
    """After step 1: walk again from the named match if the agent says wrong match and names a new one; else ircot."""
    match = str(d.get("match") or "").strip()
    if d.get("problem") == "wrong_match" and match and agent_tools.resolve(match)[0] != obs[0].seed:
        return "graph_walk", {"match": match}
    return "ircot", {}


def agent_multi(question: str):
    routes._start()
    first = agent_tools.graph_walk(question)  # step 1, fixed
    obs, replies, end = [first], [], "ladder_end"
    steps = [{"decision": {}, "fixed": True, "tool": "graph_walk", "args": first.args, "note": first.text,
              "hits": [h.chunk_id for h in first.hits]}]
    while (rung := next_rung(obs)) is not None:
        stage = {"choice": "first", "web_search": "web"}.get(rung, "plain")
        d, reply = decide(question, obs, stage)
        replies.append(reply)
        step = {"decision": {k: d.get(k) for k in ("known", "enough", "problem", "match", "missing", "web_query")
                             if k in d}}
        steps.append(step)
        if d.get("enough") is True:
            end = "enough"
            break
        if rung == "choice":
            tool, args = second_step(d, obs)
        elif rung == "web_search":
            tool, args = "web_search", {"query": str(d.get("web_query") or d.get("missing") or "").strip()}
        else:
            tool, args = rung, {}
        o = agent_tools.run(tool, args, question)
        obs.append(o)
        step |= {"tool": o.tool, "args": o.args, "note": o.text, "hits": [h.chunk_id for h in o.hits], "log": o.log}

    known = next((str(s["decision"]["known"]).strip() for s in reversed(steps) if s["decision"].get("known")), "")
    pool = build_pool(obs)
    if not pool:  # no step found a passage: GraphRAG's no-path rule, plain f6a
        pool = routes.f6a(question)
    link(question, pool, obs, known)
    tool_replies = [r for o in obs for r in o.replies]
    routes.USAGE.update(searches=len(obs), thoughts=[json.dumps(s["decision"]) for s in steps],
                        added=[[h.chunk_id for h in o.hits] for o in obs])
    LOG[question] = {"steps": steps, "tools": [o.tool for o in obs], "end": end, "cap_hit": end == "ladder_end",
                     "known": known, "web": any(o.tool == "web_search" for o in obs),
                     "pool": [h.chunk_id for h in pool], "agent_calls": len(replies)}
    try:
        return routes.answer_from(question, pool, replies + tool_replies)
    finally:
        graph_route._HOP_Q.clear()


def install(web: bool = False) -> None:
    """GraphRAG FINAL (router, f6a single, CRAG 3f + linked grading, Self-RAG), with the agent on the multi route."""
    graph_route.install(ab.SELFRAG_CONFIG, **ab.FINAL)
    WEB["on"] = web
    LOG.clear()
    baseline.ROUTE_FNS["multi"] = agent_multi
