"""Phase 4: build and check Agentic #8's fresh test set before anything is run on it. Free: text checks only, no
search, no LLM, so no system sees a test question before the one test run.

labels/new_test.jsonl (20) -> eval_harness/golden_sets/golden_cricket_agentic_test_v1.jsonl
The Self-RAG test 24 is spent (GraphRAG's test ran on it), so these are new, written from the shared library.

Kinds (from the id), the same as Self-RAG's tune and test sets:
    chain  hop 1 in a match record (player of the match / venue), hop 2 in another article    -> graph walk
    pair   two facts asked together, in two different articles                              -> multi route
    twin   a look-alike match (another Test, another innings) holds a tempting wrong answer  -> single route
    alike  a record question; look-alike "highest/lowest ... at X" lines sit in the library  -> record pages
    solo   a plain lookup
    gap    only the fallback library holds it                                               -> CRAG's fallback
    sure   the router may answer from memory, the answer is in the library
    none   no answer in any library                                                         -> must refuse

Checks, for every question:
  1. the harness loads the file (source_doc in the library, the fallback library or the record pages)
  2. no id or question repeats one in the tune set or the spent test set
  3. every evidence part is in a chunk, word for word, and the last part is in the source_doc
  4. pair / chain: two or more parts, and no single chunk holds every part
  5. chain: the first part (the match record line) is in a match article, the last in another article
  6. alike: each decoy is in a library chunk that holds no evidence part
  7. gap: no library or record chunk holds any part; the fallback library holds them all
  8. twin / solo / sure / pair / chain: the library alone holds every part (they must not need the record pages)
Where each question's evidence lives (library / records / fallback) is written out, so the record questions can be
reported apart: only #8 has the record pages.

Run from agentic_rag_in_depth/:  uv run python -I -c "import sys; sys.path.insert(0,'agentic'); import agent_golden"
"""

import json

import agentic_bridge as ab
import corpus_gap
import fallback_index
from chunks import included_docs, load_chunks
from eval_harness import load_golden
from ingest import VARIANTS
from ladder import fragments, holds, norm

LABELS = ab.ROOT / "labels"
GOLDEN_DIR = ab.RAGS / "eval_harness" / "golden_sets"
NAME = "golden_cricket_agentic_test_v1.jsonl"
SPENT = ["golden_cricket_selfrag_tune_v1.jsonl", "golden_cricket_selfrag_test_v1.jsonl"]
RESULTS = ab.RESULTS / "phase4_golden_check.json"
KINDS = ("chain", "pair", "twin", "alike", "solo", "gap", "sure", "none")


def kind(gid: str) -> str:
    return next(k for k in KINDS if gid.startswith(k + "-"))


def text(c) -> str:
    return VARIANTS["header_match"](c)  # what the answer step reads


def main():
    lines = (LABELS / "new_test.jsonl").read_text().strip().splitlines()
    (GOLDEN_DIR / NAME).write_text("\n".join(lines) + "\n")
    main_chunks = load_chunks()
    fallback = fallback_index.fallback_chunks(fallback_index.left_out())
    records = corpus_gap.record_chunks()
    record_docs = {c.doc_id for c in records}
    golden = load_golden(NAME, known_docs=set(included_docs()) | set(fallback_index.left_out()) | record_docs)
    decoys = json.loads((LABELS / "decoys.json").read_text())

    spent = [g for name in SPENT for g in load_golden(name)]
    spent_ids, spent_qs = {g.id for g in spent}, {norm(g.question) for g in spent}
    problems, report = [], {}
    for g in golden:
        k, row = kind(g.id), {"kind": kind(g.id)}
        if g.id in spent_ids:
            problems.append(f"{g.id}: id already used")
        if norm(g.question) in spent_qs:
            problems.append(f"{g.id}: question already used")
        if k == "none":
            if g.question_type != "no_answer" or g.evidence:
                problems.append(f"{g.id}: none-* must be no_answer with no evidence")
            report[g.id] = row
            print(f"  {g.id:9} none")
            continue
        parts = fragments(g)
        libs = {"library": main_chunks, "records": records, "fallback": fallback}
        where = {name: [[c for c in cs if holds(p, text(c))] for p in parts] for name, cs in libs.items()}
        row["parts"] = {name: [len(x) for x in hits] for name, hits in where.items()}
        found = [sum((where[n][i] for n in libs), []) for i in range(len(parts))]
        if not all(found):
            problems.append(f"{g.id}: an evidence part is in no chunk")
        elif not any(c.doc_id == g.source_doc for c in found[-1]):
            problems.append(f"{g.id}: the answer part is not in {g.source_doc}")
        if k in ("pair", "chain"):
            if len(parts) < 2:
                problems.append(f"{g.id}: {k} needs two evidence parts")
            if any(all(holds(p, text(c)) for p in parts) for c in main_chunks + fallback + records):
                problems.append(f"{g.id}: one chunk holds every part")
        if k == "chain":
            first = {c.doc_id for c in where["library"][0]}
            last = {c.doc_id for c in where["library"][-1]}
            row["hop1_docs"], row["hop2_docs"] = sorted(first)[:3], sorted(last)[:3]
            if not first or not last or first <= last:
                problems.append(f"{g.id}: the two hops are not in different articles")
        if k in ("twin", "solo", "sure", "pair", "chain") and not all(where["library"]):
            problems.append(f"{g.id}: the library alone does not hold every part")
        if k == "gap" and (any(where["library"]) or any(where["records"]) or not all(where["fallback"])):
            problems.append(f"{g.id}: gap, but the library or record pages hold a part, or the fallback misses one")
        if k == "alike":
            for d in decoys.get(g.id, []):
                n = sum(1 for c in main_chunks
                        if norm(d) in norm(text(c)) and not any(holds(p, text(c)) for p in parts))
                row.setdefault("decoy_chunks", []).append(n)
                if not n:
                    problems.append(f"{g.id}: decoy {d[:40]!r} is in no library chunk apart from the evidence")
            if not decoys.get(g.id):
                problems.append(f"{g.id}: alike without a decoy")
        row["evidence_in"] = sorted(n for n in libs if all(where[n]))
        report[g.id] = row
        print(f"  {g.id:9} {k:5} " + "  ".join(f"{a}={b}" for a, b in row.items() if a != "kind"))

    for p in problems:
        print(f"  PROBLEM {p}")
    counts = {k: sum(1 for g in golden if kind(g.id) == k) for k in KINDS}
    RESULTS.write_text(json.dumps({"file": NAME, "n": len(golden), "kinds": counts, "rows": report,
                                   "problems": problems}, indent=2, ensure_ascii=False))
    print(f"\n{len(golden)} questions {counts}\n{'all checks passed' if not problems else f'{len(problems)} problem(s)'}")


main()
