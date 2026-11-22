"""The agent's three tools. Each one wraps code an earlier build already has, and returns an Obs: the passages it
adds to the answer pool, and the short note the agent reads before choosing its next step.

    graph_walk(match="")    seed -> hop 1 (Neo4j link) -> hop 2 (that article's chunks) -> LLM rerank   GraphRAG #7
    ircot()                 f6a search -> one reasoning sentence -> f6a search, at most 3 searches    Adaptive #4
    web_search(query)       Google through Serper, cached (agent_web.py)                             new

graph_walk with no match is exactly GraphRAG's multi-route search (graph_route.graph_search), so its LLM rerank
replays from the cache. Its note lists the closest match names with their bge scores, so the agent can walk again
from another one when the first looks wrong. With a match the agent picked, the same walk runs from that seed.
ircot is Adaptive's search_loop without its answer step: the agent wants the passages, the answer comes later.

f6a (4 keyword rewrites + the query, dense/BM25/SPLADE, RRF, LLM rerank) is not a tool of its own: it already runs
inside graph_walk (seeds and rerank), every IRCoT search, CRAG's fallback and Self-RAG's retry. find_match and
match_facts were dropped after the smoke runs: the walk's note carries the close match names and its pool already
holds the match's record passage.

Web results are not library chunks, so each is registered as one (crag._by_id): the grader, the answer step and
Self-RAG read every passage through that table.
"""

from dataclasses import dataclass, field

import numpy as np

import agentic_bridge  # noqa: F401
import agent_web
import crag
import graph_route
import retrieve
import routes
import walk
from chunks import Chunk

CLOSE_NAMES = 3  # match names shown in the walk's note


@dataclass
class Obs:
    tool: str
    args: dict
    hits: list  # retrieve.Hit, in the order the tool ranks them
    text: str  # what the agent reads about this step (the passages are shown after it)
    log: dict = field(default_factory=dict)
    replies: list = field(default_factory=list)  # LLM replies the tool made itself (IRCoT's reasoning), for cost
    linked: dict = field(default_factory=dict)  # graph_walk: chunk id -> the question with the known hop-1 fact
    seed: str = ""  # graph_walk: the match id it started from (a second walk from it is a repeat)


# ---------- matches ----------

def top_matches(text: str, n: int = CLOSE_NAMES) -> list[tuple[str, str, float]]:
    """[(match id, name, bge score)], best first; same filter as the walk's seed (tour matches only when asked)."""
    ids, names, vecs = walk._names()
    sims = vecs @ next(retrieve._dense().query_embed(walk.ordinals(text)))
    side_ok = bool(walk.SIDE_GAMES.search(text)) or "tour match" in text.lower()
    top = [i for i in np.argsort(-sims) if side_ok or not walk.SIDE_GAMES.search(names[i])][:n]
    return [(ids[i], names[i], round(float(sims[i]), 3)) for i in top]


def resolve(match: str) -> tuple[str, str, float]:
    """The agent writes a match name; the exact one if it exists, otherwise the closest by bge."""
    ids, names, _ = walk._names()
    if match in names:
        return ids[names.index(match)], match, 1.0
    return top_matches(match, 1)[0]


# ---------- the graph walk ----------

def _walk_from(question: str, seed: tuple[str, str, float]) -> list | None:
    """GraphRAG's search with the seed fixed: walk.walk() asks seeds() for its start, so seeds() answers with ours."""
    original = walk.seeds
    walk.seeds = lambda q, fused: {"vector": None, "by_name": [seed], "by_search": []}
    try:
        return graph_route.graph_search(question)
    finally:
        walk.seeds = original


def graph_walk(question: str, match: str = "") -> Obs:
    if match:
        seed = resolve(match)
        hits = _walk_from(question, seed)
        head = f'Started from match "{seed[1]}" (your choice).'
        seed_id = seed[0]
    else:
        hits = graph_route.graph_search(question)  # GraphRAG's own multi search: cached
        top = top_matches(question)
        names = "; ".join(f'"{name}" ({score:.3f})' for _, name, score in top)
        head = f"Started from the match name closest to the question. Closest match names (similarity): {names}."
        seed_id = top[0][0]
    log = dict(graph_route.LOG.get(question, {}))
    if not hits:
        return Obs("graph_walk", {"match": match}, [], head + "\nNo path: this match's record has no link to an "
                                                            "article (venue, player of the match, top scorers).",
                   log, seed=seed_id)
    _, rel, to, line = log["via"]
    known = f"{question} (Known from the match record: in {log['via_name']}, {graph_route.HOP_FACT[rel].format(to)}.)"
    text = f'{head}\nWalked from "{log["via_name"]}" along the record line "{line}" to the article "{to}".'
    return Obs("graph_walk", {"match": match}, hits, text, log, linked={c: known for c in log["chunks"]},
               seed=seed_id)


# ---------- the search loop ----------

def ircot(question: str) -> Obs:
    """Adaptive's search_loop (search="f6a"), stopped before its answer step. Same prompts, so cached replays."""
    pool, thoughts, replies, searches = routes.f6a(question), [], [], [question]
    while True:
        _, context = routes.numbered(pool)
        reply, query = routes.ircot_step(question, context, thoughts, searches)
        replies.append(reply)
        thoughts.append(reply.text)
        if query is None or len(searches) == routes.MAX_SEARCHES:
            break
        seen = {h.chunk_id for h in pool}
        pool += [h for h in routes.f6a(query) if h.chunk_id not in seen]
        searches.append(query)
    text = f"Search-reason-search loop ({len(searches)} searches). Its reasoning: " + " ".join(thoughts)
    return Obs("ircot", {}, pool, text, {"searches": searches, "thoughts": thoughts}, replies)


# ---------- the web ----------

def web_search(question: str, query: str = "") -> Obs:
    q = query or question
    hits = []
    for n, r in enumerate(agent_web.search(q)):
        chunk = Chunk(chunk_id=f"web::{agent_web.key(q)}::{n}", doc_id="web", path=("Web", r["title"]),
                      text=f"{r['snippet']}\n({r['link']})", n_tokens=0)
        crag._by_id[chunk.chunk_id] = chunk
        hits.append(retrieve.Hit(chunk.chunk_id, chunk.doc_id, list(chunk.path), chunk.text, 0.0))
    return Obs("web_search", {"query": query}, hits, f'Web search for "{q}".', {"query": q})


# name -> (function, the argument the agent fills in, or None)
TOOLS = {"graph_walk": (graph_walk, "match"), "ircot": (ircot, None), "web_search": (web_search, "query")}


def run(tool: str, args: dict, question: str) -> Obs:
    fn, arg = TOOLS[tool]
    return fn(question, str(args.get(arg) or "")) if arg else fn(question)
