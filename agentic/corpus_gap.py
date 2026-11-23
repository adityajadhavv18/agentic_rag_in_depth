"""The corpus-gap check (corpus_gap_log.md): add Wikipedia's record pages to the library, for this build only.

    uv run agentic/corpus_gap.py fetch     # pin each page's current revision, download its HTML        free
    uv run agentic/corpus_gap.py build     # clean + chunk like the library; sizes and the notes' cost   free
    uv run agentic/corpus_gap.py notes     # the library's LLM note for every new chunk                 PAID, once
    uv run agentic/corpus_gap.py index     # Qdrant: the library's collection copied + the new chunks   free
    uv run agentic/corpus_gap.py check A_records_selfrag   # 78 tune questions without web (RUNS)  PAID

Nothing shared changes: the pages live in data/records_pages/ (raw HTML, text, manifest with pinned revids, notes),
and the index is a NEW collection, cricket_header_match_note_records: every point of Contextual's
cricket_header_match_note copied as it is (same vectors), plus the record chunks. install(True) points this build's
search at it (f6a's lists, the walk's hop-2 ranking, Self-RAG's retry, CRAG's deep search); the fallback library,
the graph and every other build keep the old collection.

Built exactly like the library (Contextual #2): Contextual's own cleaner (clean_wiki.clean) and chunker (300 bge
tokens, 50 overlap, inside one section, title › heading labels), the same note prompt (notes.PROMPT, section mode
unless it would cost too much, then chunk mode), the same three vectors, and BM25 with the library's average length,
so the old chunks' keyword weights stay comparable. Downloaded HTML is untrusted: it is only parsed as text.
"""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from functools import cache

import requests

import agentic_bridge as ab

sys.path.append(str(ab.RAGS / "contextual_rag_in_depth" / "scripts"))  # clean_wiki, build_manifest (bare names)

import crag  # noqa: E402
import ingest  # noqa: E402
import notes  # noqa: E402
import rag_fusion  # noqa: E402
import walk  # noqa: E402
from chunks import EMBED_MODEL, Chunk, boxes, chunk_section, load_chunks, sections  # noqa: E402
from eval_harness import settings, traced_openai  # noqa: E402

DIR = ab.ROOT / "data" / "records_pages"
RAW, TEXT = DIR / "raw", DIR / "text"
MANIFEST, NOTES = DIR / "manifest.json", DIR / "notes.json"
REC = "cricket_header_match_note_records"
SOURCE = rag_fusion.NOTE  # "cricket_header_match_note"
PAGES = [  # one records page per format and per tournament the library covers (corpus_gap_log.md)
    "List of Indian Premier League records and statistics",
    "List of Test cricket records",
    "List of One Day International cricket records",
    "List of Twenty20 International records",
    "List of Cricket World Cup records",
    "List of ICC Men's T20 World Cup records",
]  # "List of ICC Champions Trophy records" redirects to the tournament's main article, not a records page: left out
API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "rag-research-client/0.1 (non-commercial research)"
SECTION_MODE_MAX_USD = 0.10  # above this, the notes read the chunk's label only (Contextual's cheaper chunk mode)
_last = [0.0]


def api(**params) -> dict:
    """One polite Wikipedia API call (~1 a second)."""
    params |= {"format": "json", "formatversion": 2}
    for attempt in range(5):
        time.sleep(max(0.0, _last[0] + 1.0 - time.time()))
        _last[0] = time.time()
        r = requests.get(API, params=params, headers={"User-Agent": USER_AGENT}, timeout=60)
        if r.status_code in (429, 503):
            time.sleep(5 * (attempt + 1))
            continue
        r.raise_for_status()
        data = r.json()
        if "error" in data:
            raise RuntimeError(f"Wikipedia API error: {data['error']}")
        return data
    raise RuntimeError("Wikipedia API kept throttling")


def manifest() -> list[dict]:
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else []


# ---------- fetch ----------

def fetch() -> None:
    from build_manifest import doc_id
    RAW.mkdir(parents=True, exist_ok=True)
    pinned = {e["asked"]: e for e in manifest()}
    for title in PAGES:
        if title not in pinned:  # pin once: later runs read the same revision
            page = api(action="query", titles=title, redirects=1, prop="revisions", rvprop="ids|timestamp")
            page = page["query"]["pages"][0]
            if page.get("missing"):
                print(f"  missing on Wikipedia: {title}")
                continue
            rev = page["revisions"][0]
            pinned[title] = {"asked": title, "title": page["title"], "doc_id": doc_id(page["title"]),
                             "revid": rev["revid"], "timestamp": rev["timestamp"]}
        e = pinned[title]
        path = RAW / f"{e['doc_id']}.html"
        if not path.exists():
            data = api(action="parse", oldid=e["revid"], prop="text", disableeditsection=1, disabletoc=1)
            path.write_text(data["parse"]["text"])
        print(f"  {e['title']}  (revision {e['revid']}, {e['timestamp'][:10]})")
    MANIFEST.write_text(json.dumps(list(pinned.values()), indent=1, ensure_ascii=False) + "\n")


# ---------- build ----------

def build() -> list[Chunk]:
    from clean_wiki import clean
    TEXT.mkdir(parents=True, exist_ok=True)
    for e in manifest():
        (TEXT / f"{e['doc_id']}.txt").write_text(clean((RAW / f"{e['doc_id']}.html").read_text(), e["title"]))
    record_chunks.cache_clear()
    return record_chunks()


@cache
def record_chunks() -> list[Chunk]:
    """Chunked exactly as chunks.load_chunks() chunks a library article."""
    out = []
    for e in manifest():
        secs = sections((TEXT / f"{e['doc_id']}.txt").read_text())
        last = None
        for n, (path, body) in enumerate(secs):
            found = boxes(body, path[0])
            inherited = last[1] if last and path[:len(last[0])] == last[0] else None
            out.extend(chunk_section(e["doc_id"], n, path, body, found, inherited))
            if found:
                last = (path, found[-1][1])
            elif inherited is None:
                last = None
    return out


@cache
def section_of() -> dict[str, str]:
    bodies = {}
    for e in manifest():
        secs = sections((TEXT / f"{e['doc_id']}.txt").read_text())
        for c in record_chunks():
            if c.doc_id == e["doc_id"]:
                bodies[c.chunk_id] = secs[int(c.chunk_id.split("::")[1][1:])][1]
    return bodies


def prompt(c: Chunk, mode: str) -> str:
    section = notes.SECTION.format(body=section_of()[c.chunk_id]) if mode == "section" else ""
    return notes.PROMPT.format(label=notes.label(c), section=section, chunk=c.text)


def estimate(chunks: list[Chunk]) -> dict[str, float]:
    """Upper bound of the notes' cost per mode: ~4 characters a token, ~60 tokens a note."""
    out = {}
    for mode in ("section", "chunk"):
        tokens_in = sum(len(prompt(c, mode)) for c in chunks) / 4
        out[mode] = notes.cost({"in": tokens_in, "cached": 0, "out": 60 * len(chunks)})
    return out


def report_build(chunks: list[Chunk]) -> None:
    for e in manifest():
        mine = [c for c in chunks if c.doc_id == e["doc_id"]]
        words = len((TEXT / f"{e['doc_id']}.txt").read_text().split())
        print(f"  {e['title'][:58]:58s} {words:7,d} words  {len(mine):4d} chunks")
    est = estimate(chunks)
    print(f"  total {len(chunks)} chunks; notes cost about ${est['section']:.3f} (section mode) or "
          f"${est['chunk']:.3f} (chunk mode)")


# ---------- notes ----------

def write_notes(workers: int = 8) -> None:
    chunks = record_chunks()
    est = estimate(chunks)
    mode = "section" if est["section"] <= SECTION_MODE_MAX_USD else "chunk"
    store = json.loads(NOTES.read_text()) if NOTES.exists() else {}
    todo = [c for c in chunks if store.get(c.chunk_id, {}).get("chunk_start") != c.text[:80]]
    print(f"  {len(todo)} notes to write, {mode} mode (estimate ${est[mode]:.3f})", flush=True)
    client, usage = traced_openai(), {"in": 0, "cached": 0, "out": 0, "calls": 0}

    def write(c: Chunk):
        return c, client.chat.completions.create(model=settings.llm_model, temperature=0, max_tokens=100,
                                                 messages=[{"role": "user", "content": prompt(c, mode)}])

    save = lambda: NOTES.write_text(json.dumps(store, indent=1, ensure_ascii=False))
    with ThreadPoolExecutor(workers) as pool:
        for c, resp in pool.map(write, todo):
            u = resp.usage
            cached = u.prompt_tokens_details.cached_tokens if u.prompt_tokens_details else 0
            store[c.chunk_id] = {"note": resp.choices[0].message.content.strip(), "chunk_start": c.text[:80],
                                 "mode": mode}
            for key, n in (("in", u.prompt_tokens), ("cached", cached), ("out", u.completion_tokens), ("calls", 1)):
                usage[key] += n
            if usage["calls"] % 50 == 0:
                save()
    save()
    print(f"  wrote {usage['calls']} notes, real cost ${notes.cost(usage):.4f}")


# ---------- index ----------

def index_text(c: Chunk, note: str) -> str:
    return "\n".join([" › ".join(c.path), *c.match, note, c.text])  # the header_match_note variant


def index() -> None:
    from fastembed import SparseTextEmbedding, TextEmbedding
    from qdrant_client import QdrantClient, models
    client = QdrantClient(url=ingest.QDRANT_URL, timeout=300)
    params = client.get_collection(SOURCE).config.params
    if client.collection_exists(REC):
        client.delete_collection(REC)
    client.create_collection(REC, vectors_config=params.vectors, sparse_vectors_config=params.sparse_vectors)
    offset, copied = None, 0
    while True:  # the library, point for point, vectors unchanged
        points, offset = client.scroll(SOURCE, limit=256, offset=offset, with_vectors=True, with_payload=True)
        client.upsert(REC, points=[models.PointStruct(id=p.id, vector=p.vector, payload=p.payload) for p in points])
        copied += len(points)
        if offset is None:
            break
    print(f"  copied {copied} library points", flush=True)

    store = json.loads(NOTES.read_text())
    chunks = record_chunks()
    texts = [index_text(c, store[c.chunk_id]["note"]) for c in chunks]
    library = [ingest.VARIANTS["header_match_note"](c) for c in load_chunks()]
    avg = ingest.avg_words(library)  # BM25's length normalisation, as the library was indexed with
    dense = list(TextEmbedding(EMBED_MODEL).passage_embed(texts, batch_size=8))
    bm25 = list(SparseTextEmbedding(ingest.BM25_MODEL, avg_len=avg).embed(texts))
    splade = list(SparseTextEmbedding(ingest.SPLADE_MODEL).embed(texts, batch_size=8))
    client.upload_points(REC, points=(models.PointStruct(
        id=ingest.point_id(c.chunk_id),
        vector={ingest.DENSE: dense[i].tolist(), ingest.BM25: ingest.sparse(bm25[i]),
                ingest.SPLADE: ingest.sparse(splade[i])},
        payload={"chunk_id": c.chunk_id, "doc_id": c.doc_id, "path": list(c.path), "text": c.text})
        for i, c in enumerate(chunks)), batch_size=32, wait=True)
    print(f"  '{REC}': {client.count(REC, exact=True).count} points ({copied} library + {len(chunks)} records)")


# ---------- switch ----------

def install(on: bool = True) -> None:
    """Point this build's searches at the records collection (on) or back at the library's (off)."""
    coll = REC if on else SOURCE
    rag_fusion.FUSION["f5a"][1]["collection"] = coll  # f6a everywhere: single route, IRCoT, walk seeds, retry
    walk.NOTE = coll  # the walk's hop-2 ranking
    if on:
        for c in record_chunks():  # the grader, answer step and Self-RAG read passages through this table
            crag._by_id[c.chunk_id] = c


# ---------- check ----------

RUNS = {  # name -> (records on, Self-RAG on, web on, the run it is compared with)
    "A_records_selfrag": (True, True, False, "phase3b_tune_noweb"),  # the clean test: only the corpus changes
    "B_norecords_noselfrag": (False, False, False, "phase3b_tune_noweb"),  # Self-RAG removed, old library (replay)
    "C_records_noselfrag": (True, False, False, "phase3b_tune_noweb"),  # records on, Self-RAG removed
    "D_records_noselfrag_web": (True, False, True, "corpus_gap_C_records_noselfrag"),  # C + web (ladder, both routes)
}


def check(name: str) -> None:
    from eval_harness import load_golden
    import agent_loop
    import agent_smoke
    import agent_tune
    import baseline
    import route_eval
    import selfrag_golden as sg
    records, selfrag, web, ref = RUNS[name]
    golden = {g.id: g for g in load_golden(sg.golden_name("tune"))}
    base = json.loads((ab.RESULTS / "phase1.json").read_text())["questions"]
    prev = json.loads((ab.RESULTS / f"{ref}.json").read_text())["questions"]
    before = agent_smoke.snapshot()
    agent_loop.install(web=web, selfrag=selfrag)
    install(records)
    record_docs = {e["doc_id"] for e in manifest()}
    picks = {}
    print(f"=== {name}: records {'on' if records else 'off'}, Self-RAG {'on' if selfrag else 'off'}, "
          f"web {'on' if web else 'off'}",
          flush=True)
    rows = route_eval.run_route(f"agent_{name}", golden, baseline.routed(picks), golden_set=sg.golden_name("tune"))
    for i, row in rows.items():  # the same fields agent_tune.run() keeps
        q = golden[i].question
        log = agent_tune.self_rag.LOG.get(q, {})
        row["route"], row["kind"] = picks[q], sg.kind(i)
        row["correct3"] = agent_tune.answer_tuning.grade3(golden[i], row["answer"])
        row["crag"] = crag.LOG.get(q)
        row["selfrag"] = {k: log.get(k) for k in ("sure", "picked", "good", "retries", "candidates")}
        row["agent"] = agent_loop.LOG.get(q) if picks[q] == "multi" else None
        row["single_web"] = agent_loop.SINGLE_LOG.get(q) if picks[q] == "single" else None
        row["record_passages"] = [c for c in (row["crag"] or {}).get("kept", []) if c.split("::")[0] in record_docs]
    s = agent_tune.report(rows, base, golden, prev)
    s["spend_usd"] = round(agent_smoke.spent(before), 4)
    s["used_records"] = [i for i, r in rows.items() if r["record_passages"]]
    s["setup"] = {"records": records, "selfrag": selfrag, "web": web, "compared_with": ref}
    s["web_ran_single"] = [i for i, r in rows.items() if (r["single_web"] or {}).get("web")]
    s["no_answer_broken_vs_ref"] = [i for i in rows if golden[i].question_type == "no_answer"
                                    and rows[i]["correct"] < prev[i]["correct"]]
    (ab.RESULTS / f"corpus_gap_{name}.json").write_text(json.dumps({"summary": s, "questions": rows}, indent=2,
                                                                    ensure_ascii=False))
    print(f"right (78): {s['correct']:.3f} (1 vote), {s['correct3']:.3f} (3 votes); {ref}: "
          f"{sum(p['correct'] for p in prev.values()) / 78:.3f}")
    print(f"changed vs that: {s['vs_phase3'] or 'none'}")
    print(f"record passages kept by CRAG on: {s['used_records'] or 'none'}")
    if web:
        print(f"web ran on single: {s['web_ran_single'] or 'none'}; on multi: {s['web_ran'] or 'none'}; "
              f"no_answer broken: {s['no_answer_broken_vs_ref'] or 'none'}")
    for i in ("alike-04", "alike-05", "trap-05"):
        print(f"  {i}: {prev[i]['correct']:g} -> {rows[i]['correct']:g} | {rows[i]['answer'][:140]}")
    print(f"$/question {s['cost_per_question']:.5f}; p50 {s['p50_ms'] / 1000:.1f} s; multi p50 "
          f"{s['multi_p50_ms']['agent'] / 1000:.1f} s; real spend ${s['spend_usd']:.4f}", flush=True)


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "build"
    if step == "fetch":
        fetch()
    elif step == "build":
        report_build(build())
    elif step == "notes":
        write_notes()
    elif step == "index":
        index()
    elif step == "check":
        check(sys.argv[2])
