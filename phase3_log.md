# Phase 2–3 log: problems and the fixes I made

Every number is on the 78-question tune set (1-vote grader; the 3-vote grader agreed in every final run). The base is
GraphRAG #7 FINAL, replayed at 0.942. The test set has not been run.

## Where it ended

| Setup | Tune | $/question | Median time |
|---|---|---|---|
| GraphRAG #7 FINAL | 0.942 | $0.0034 | 11.2 s |
| Agent ladder (Phase 3, Self-RAG on, no web) | 0.949 | $0.0049 | 12.4 s |
| **Agent ladder + record pages, Self-RAG off, no web (final, run C)** | **0.974** | **$0.0029** | **9.3 s** |
| Same + web as a last resort (run D) | 0.962 | $0.0030 | 11.0 s |

The final system: router → none (memory, "are you sure?") / single (f6a) / multi (graph walk → agent: enough? →
IRCoT) → CRAG 3f with linked grading → answer, over the library plus six Wikipedia record pages.

## Problems and fixes, in order

### 1. A free-choosing agent picked tools badly
- **Problem.** With six tools (graph walk, IRCoT, f6a search, find match, match facts, web) and a cap of 3 steps,
  the agent opened every two-hop question with two lookups (find_match → match_facts) and hit the cap before it had
  the evidence. Smoke: 3/5 against GraphRAG's 4.5/5.
- **First fix, worse.** Lookups stopped counting as steps, and the agent wrote down the facts it knew so the grader
  could use them. The new menu layout changed its strategy: it used f6a for everything and never the graph. 2/5.
- **Fix.** Three tools only (graph walk, IRCoT, web) and a fixed ladder: the graph walk always runs first, then
  the agent decides only "enough?". Not enough → IRCoT → (web). Smoke: 5/5. gpt-4o-mini decides "is this enough?"
  far more reliably than it picks tools.
- **Dropped tools.** f6a stays as the search engine inside the walk, IRCoT, CRAG and Self-RAG, but not as a tool:
  IRCoT and Self-RAG's retry already write their own queries. find_match's job moved into the walk's note (the three
  closest match names); match_facts duplicated the match record passage the walk already brings.

### 2. The web tool was on the menu before the code accepted it
- **Problem.** The prompt listed web_search from step 1; the code accepted it only after a library search.
- **Fix.** The web line appears only when the web may run.

### 3. Web results could be cut out of the pool
- **Problem.** The pool holds 15 passages. A walk (up to 9) plus IRCoT (up to 13) filled it, and web, read last,
  was dropped before CRAG saw it.
- **Fix.** When the web runs, its top 3 go first (the ladder reaches it only after the library fell short), then
  the walk's passages, then IRCoT's. Every step keeps at least 3 seats. Checked end to end: 11/11.

### 4. The agent's own "known facts" misled the grader
- **Problem.** Grading IRCoT and web passages against the facts the agent said it knew fixed one smoke question,
  but the agent's facts were wrong three times (chain-06, twin-03, gap-02). On gap-02 a wrong fact made CRAG accept
  the wrong passages, and the answer became "I don't know".
- **Fix.** Removed. Only graph passages keep linked grading, because their link comes from a record line in Neo4j,
  not from the LLM. chain-06's gain held without it.

### 5. The second graph walk never ran
- **Problem.** The agent could say "wrong match" and name another match to walk from. On tune it did so twice, both
  times naming the match it had already walked from.
- **Fix.** Removed. IRCoT already recovers from a wrong starting match (chain-06), and the prompt became simpler.

### 6. Web search on every single-route question broke right answers
- **Problem.** Compulsory web fixed one question (alike-04) and broke three: a true fact about another innings
  (Archer 6/45 for pron-01), a snippet too short to answer (gap-03), and web passages that graded well enough to
  stop CRAG from searching its fallback library, which held the answer (gap-01, gap-03).
- **Fix.** The single route got the multi ladder's rule: f6a → agent: enough? → only then the web.

### 7. Self-RAG's critic rejected right answers and added nothing
- **Problem.** When the web found the answer ("24 August 2025", "Gayle 175*"), the critic said no excerpt was about
  the question and picked "I don't know". An offline check over every saved run showed its picks matched CRAG's own
  answer on all but two questions without web, with no score change.
- **Fix.** Self-RAG's candidates, critic and retry are removed; its "are you sure?" step on the none route stays.
  Same accuracy (0.949 → 0.949, a cached replay), about 40% cheaper and a third faster. This matches what #6 found
  when I built it (0.910 → 0.897 with gpt-4o-mini as the critic).

### 8. Simple record questions failed because the library had no record pages
- **Problem.** "Highest individual score in an IPL match" and "best IPL bowling figures" failed in every build. The
  library held no records page; the answers appeared only in passing, in other words than the question's, behind
  look-alikes that fooled CRAG.
- **Fix.** Six Wikipedia record pages added, for this build only, in a separate collection: 406 chunks, built
  exactly like the library. Both questions fixed, nothing broken: 0.949 → 0.974. Details: corpus_gap_log.md.

### 9. With the record pages in, the web added nothing
- **Problem checked.** Run D put the web back as a last resort on both routes. The agent sent 19 of 30 single
  questions to the web and gained nothing; one answer changed (trap-02), because the web is newer than the library:
  it reports a 24-ball debut fifty from March 2025 that beats the golden answer's 26-ball record.
- **Decision.** No web in the final setup; D is reported as what the web adds (0.962, or 0.974 counting trap-02).

## Still open

- **trap-05** (highest Test score by an Indian against England: Nair 303*). The library's only "highest" sentence
  is about Kohli's 235, written a week before Nair passed it. The agent saw both numbers and doubted it ("unclear
  which is the highest"), but the answer step trusted the stated claim over comparing numbers.
- **none-11** (should be "I don't know"): a 2023 passage with Steve Smith's 121 passes for the 2021 final.
- **Latency on multi**: the agent climbs to IRCoT on about half the multi questions (median 18 s against
  GraphRAG's 12 s).

## Spend

About $0.54 of gpt-4o-mini across Phases 2–3, including the record pages' notes ($0.065). About 50 Serper searches.
