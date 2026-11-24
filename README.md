# Agentic RAG #8: cricket on Wikipedia

GraphRAG #7 won on tune and lost on test. Its graph walk is great when it starts from the right match and has no
second chance when it doesn't: on one test question it took "the 2025 Champions Trophy final" for "the 2025 IPL
final" and followed the wrong match to a confident "I don't know". IRCoT, the search-and-reason loop it replaced,
was slower but could recover. So this build gives an agent **both**, plus a web search, and lets it decide when the
evidence is enough.

The agent I ended up with is much simpler than the one I set out to build. Letting gpt-4o-mini choose among tools
failed twice, so it climbs a **fixed ladder** and decides only one thing: "is this enough?". Along the way the build
found that the biggest gains on its questions came not from the agent but from **removing a checker** and **adding
six pages the library never had**.

| | Tune (78) | Test (20, unseen) |
|---|---|---|
| GraphRAG #7 FINAL | 0.942 | 0.825 |
| **Agentic #8 FINAL** (ladder agent + record pages, Self-RAG critic off, no web) | **0.974** | **0.975** |
| the same, without the 3 record questions (fair to #7) | — | 0.912 → **0.971** |
| multi-step questions | 37.5 → 38 of 39 | 12.5 → 12.5 of 13 |
| $ per question | $0.0034 → $0.0029 | $0.0043 → $0.0029 |
| median time | 11.2 s → 9.3 s | 17.0 s → 9.0 s |
| + web search as a last resort | 0.962 | 0.925 |

**On unseen questions the final system beat GraphRAG by 3 questions in 20, about a third cheaper and twice as
fast.** Where the gains came from matters more than the score: one came from removing Self-RAG's critic (it picked
a wrong candidate over CRAG's right answer), two from the record pages. On multi-step questions, the agent's own
territory, it tied GraphRAG on test (12.5 of 13 each) and gained half a question on tune. Web search never gained
a question on either set, so the final system has none.

## Architecture

```
                          ┌──────────────────────┐
                          │   User's question    │
                          └──────────┬───────────┘
                                     ▼
                          ┌──────────────────────┐
                          │ ROUTER (Adaptive #4) │   few-shot gpt-4o-mini, one word; unchanged
                          └──┬────────┬────────┬─┘
             ┌───────────────┘        │        └─────────────────────┐
             ▼                        ▼                              ▼
  ┌──────────────────┐   ┌───────────────────┐   ┌───────────────────────────────────────┐
  │ NONE             │   │ SINGLE            │   │ MULTI: THE AGENT'S LADDER (new)       │
  │ memory + "are    │   │ f6a search        │   │                                       │
  │ you sure?"       │   │ (Fusion #3):      │   │ 1 GRAPH WALK  (GraphRAG #7, always)   │
  │ (Self-RAG #6)    │   │ 4 rewrites + the  │   │      │                                │
  │ unsure → SINGLE ─┼──▶│ question; dense / │   │      ▼                                │
  └────────┬─────────┘   │ BM25 / SPLADE;    │   │   AGENT: "enough?" (gpt-4o-mini, JSON)│
           │             │ RRF; LLM rerank   │   │      │ yes ──────────────► pool       │
           │             │ → top 5           │   │      ▼ no                             │
           │             └─────────┬─────────┘   │ 2 IRCoT  (Adaptive #4: search →       │
           │                       │             │   reason → search, ≤ 3)  ──► pool     │
           │                       │             │                                       │
           │                       │             │ POOL  walk's passages first, then     │
           │                       │             │       IRCoT's; no repeats; ≤ 15       │
           │                       │             └───────────────────┬───────────────────┘
           │                       └──────────────┬──────────────────┘
           │                                      ▼
           │        ┌───────────────────────────────────────────────────────────┐
           │        │ SEARCH INDEX: 140 articles (4,650 chunks)                 │
           │        │             + 6 Wikipedia record pages (406 chunks, new)  │
           │        └───────────────────────────────────────────────────────────┘
           │                                      ▼
           │        ┌───────────────────────────────────────────────────────────┐
           │        │ CRAG (Corrective #5): grade each passage 10 / 5 / 0;      │
           │        │ not CORRECT → fallback library of 9 biographies           │
           │        │ + LINKED GRADING (GraphRAG #7): a graph passage is graded │
           │        │   with the record line that brought it                    │
           │        └─────────────────────────────┬─────────────────────────────┘
           │                                      ▼
           │        ┌───────────────────────────────────────────────────────────┐
           │        │ ANSWER from CRAG's passages (Self-RAG's candidates, critic│
           │        │ and retry removed)                                        │
           │        └─────────────────────────────┬─────────────────────────────┘
           ▼                                      ▼
                          ┌──────────────────────┐
                          │        ANSWER        │
                          └──────────────────────┘
```

Think of a junior doctor with a patient. She always reads the chart first (the graph walk: cheap, and usually
enough). If the chart doesn't settle it, she orders tests (IRCoT: slower, but it reasons its way to what's
missing). A specialist (the web) is the last call. What she must never do is skip the chart because she has a
hunch about which test to run, and that is exactly what the free-choosing agent did.

- **ReAct** (Yao et al., 2023, "ReAct: Synergizing Reasoning and Acting in Language Models"): the think → act →
  observe loop. My agent observes after every step, but its only "act" decision is whether to stop.
- **Agentic RAG** (Singh et al., 2025, "Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG"): agents
  that plan retrieval and pick tools. The planning part is what failed here with a small model.
- **IRCoT** (Trivedi et al., 2023) and **GraphRAG** (Edge et al., 2024): the agent's two library tools, reused from
  Adaptive #4 and GraphRAG #7.
- **CRAG** (Yan et al., 2024): the passage grader and its fallback library, unchanged.
- **Self-RAG** (Asai et al., 2023): built in #6; its "are you sure?" step stays, its critic goes (Phase 3).

## Same system, one route changed

The base is GraphRAG #7's final system, reused as it is through `agentic/agentic_bridge.py`: the 140 cricket
Wikipedia articles and 4,650 chunks of [contextual-rag-in-depth](https://github.com/adityajadhavv18/contextual-rag-in-depth),
Adaptive #4's router, Fusion #3's f6a search, CRAG with its 9-biography fallback library, GraphRAG's walk over the
Neo4j cricket graph with linked grading, and Self-RAG's layer. The agent replaces one entry in the router's table,
the multi route. Two later changes, both measured on their own, also touched the rest: the record pages (the search
index) and removing Self-RAG's critic (the answer step).

**Rulers:**
- **Tune:** Self-RAG's 78 questions, the same set #6 and #7 were tuned on. The router sends 39 to multi.
- **Test:** 20 fresh questions written for this build (Self-RAG's 24 were spent by GraphRAG's test). Details in
  [Test](#test-20-fresh-questions-run-once).

Scores are the share answered right, marked by Contextual #2's answer checker (1 / 0.5 / 0), with a 3-vote checker
beside it. "I don't know" is 0 on an answerable question and right on a no-answer one.

## The build, problem by problem

### Phase 1: the reuse works

GraphRAG #7's final system, through the bridge, on the 78 tune questions: **0.942** (3 votes 0.936), every question
identical to #7's saved run, at $0 (a dummy API key proved every call came from the cache).

### Phase 2: three agents, one that worked

**A free check first.** The plan was to let the agent spot a shaky walk from its seed confidence. On the 39 multi
questions, 20 had two candidate matches within 0.005 similarity, yet GraphRAG was wrong on only 2 of them. Seed
confidence is a weak cue, so the agent gets to see the walk's result, not just its score.

**Agent 1: free choice from six tools** (graph walk, IRCoT, f6a search, find match, match facts, web), up to 3
steps. On 5 smoke questions it scored 3/5 against GraphRAG's 4.5/5. It opened almost every question with two
lookups (find the match, read its facts) and ran out of steps before it had any evidence.

**Agent 2: the same, with lookups free and the agent writing down what it knew** so the grader could use it. It
got worse, 2/5. The new menu layout changed its whole strategy: it used plain f6a search for everything and never
touched the graph. And the facts it "knew" were wrong (it put chain-06's Test at the MCG).

**Agent 3: a ladder.** Three tools, a fixed order, and one decision:

```
  AGENT 1 (free choice, 6 tools)        AGENT 3 (the ladder, 3 tools)

  question                              question
     │                                     │
     ▼                                     ▼
  pick a tool ◄──────┐                  graph walk            (always first)
     │               │                     │
     ▼               │                     ▼
  find_match         │                  enough? ── yes ──► answer
  match_facts        │ ≤ 3 steps           │ no
  f6a / walk / ...   │                     ▼
     │               │                  IRCoT                 (always second)
     ▼               │                     │
  observe ───────────┘                     ▼
     │                                  [enough? ── yes ──► answer]   web mode only
     ▼                                     │ no
  answer (often before                     ▼
  any evidence arrived)                 web search            (last resort, web mode only)
                                           │
                                           ▼
                                        answer
```

5/5 on the smoke, in both modes. **gpt-4o-mini judges "is this enough?" far more reliably than it plans.** Tool
choice swung with the layout of the prompt; a yes/no on evidence it can read did not.

Cutting from six tools to three:
- **f6a search** stays as the engine inside the walk, IRCoT and CRAG, but not as a tool: IRCoT already writes its
  own queries, so a separate search step only duplicated it.
- **find match** moved into the walk's note (the three closest match names, with similarity).
- **match facts** duplicated the match-record passage the walk already brings.

### Phase 3: tuning the ladder

The first tune run with the ladder, Self-RAG on, no web: **0.949** against GraphRAG's 0.942. One question changed:
chain-06 ("in what year was the first Test played at the ground that hosted the second Test of India's 2024–25
tour of Australia?"). The walk started from the wrong Test; the agent said "not enough", IRCoT found Adelaide, 1884.
But a run with web on scored 0.936, and fixing that led through six problems.

| # | Problem | Fix |
|---|---|---|
| 1 | The web tool was on the agent's menu before the code allowed it | The web line appears only when the web may run |
| 2 | The pool holds 15 passages. Walk (≤ 9) + IRCoT (≤ 13) filled it, so web results, read last, were cut before CRAG saw them | When the web runs, its top 3 go first; every step keeps at least 3 seats |
| 3 | The agent's own "known facts" were used to grade IRCoT and web passages. They were wrong three times; on gap-02 a wrong fact made CRAG accept wrong passages | Removed. Only graph passages get linked grading, because their link comes from a record line in Neo4j, not from the LLM |
| 4 | A "walk again from another match" rung never helped: both times the agent named the match it had just walked from | Removed. IRCoT already recovers from a wrong start (chain-06) |
| 5 | Web search on every single-route question fixed 1 and broke 3: a true fact about another innings, a snippet too short to answer, and web passages good enough to stop CRAG from opening its fallback library, which held the answer | The single route got the multi ladder's rule: library first, agent asks "enough?", only then the web |
| 6 | Self-RAG's critic rejected right web answers ("24 August 2025", "Gayle 175*") and picked "I don't know" | An offline check over every saved run: without web the critic's pick matched CRAG's own answer on all but 2 questions, with no score change. Removed (candidates, critic, retry); "are you sure?" stays. 0.949 → 0.949, 40% cheaper, a third faster |

Fix 6 matches what Self-RAG #6 itself found (0.910 → 0.897 when the critic was added): with gpt-4o-mini as the
judge, the critic costs more than it catches. On the test set it cost a question (below).

### Phase 3: the corpus gap

"What is the highest individual score in an IPL match, and who made it?" is about the simplest IPL question there
is (Chris Gayle, 175*). Self-RAG #6, GraphRAG #7 and the agent all got it wrong. So did "best bowling figures in IPL
history". The reason was not the retrieval:

- **The library had no record pages.** 140 articles (every IPL season, the Ashes, World Cups, grounds, players), not
  one records or statistics page.
- **The answers appeared only in passing, in other words.** Gayle's record is one sentence of the 2013 IPL article:
  "the highest in **Twenty20 cricket**". The question says "IPL".
- **Look-alikes won and fooled the safety net.** Dozens of passages say "highest score at this ground" or "highest
  this season". CRAG graded one of them 10, said CORRECT, and so neither the fallback library nor the web ever ran.

I added the six record pages that match the library's scope (IPL, Test, ODI, T20I, Cricket World Cup, Men's T20
World Cup; the Champions Trophy has none), chosen by scope, not by the failing questions. They were fetched at
pinned revisions and chunked, noted and indexed exactly like the library, in a separate collection so builds #1 to
#7 keep their corpus.

```
  BEFORE                                        AFTER
  "highest individual score in an IPL match?"   "highest individual score in an IPL match?"
     │                                             │
     ▼                                             ▼
  top 5: highest at Eden Gardens (Patidar 112)  top 5: IPL records page › Highest individual
         highest this season (Gill 129) ...            score: 175* Chris Gayle ... (ranked 1st)
     │                                             │
     ▼                                             ▼
  CRAG: 10 → CORRECT → no fallback, no web      CRAG: CORRECT
     │                                             │
     ▼                                             ▼
  "Rajat Patidar, 112"                ✗         "175*, made by Chris Gayle"        ✓
```

| Setup on tune (no web) | Tune | Changed | $/question | Median |
|---|---|---|---|---|
| Agent, old library, Self-RAG on | 0.949 | — | $0.0049 | 12.4 s |
| Agent, old library, Self-RAG off | 0.949 | none | $0.0029 | 8.9 s |
| Agent + record pages, Self-RAG on | 0.974 | alike-04, alike-05 0 → 1 | $0.0048 | 14.9 s |
| **Agent + record pages, Self-RAG off (FINAL)** | **0.974** | alike-04, alike-05 0 → 1 | **$0.0029** | **9.3 s** |
| + web as a last resort | 0.962 | trap-02 1 → 0 | $0.0030 | 11.0 s |

Both record questions were fixed and nothing broke; CRAG kept record passages on 9 questions and none got worse.
Eight architectures had not fixed what six pages fixed. The full write-up is in [corpus_gap_log.md](corpus_gap_log.md).

**With the record pages in, the web added nothing.** The agent sent 19 of 30 single questions and 9 multi questions
to Google and gained none. The one change, trap-02 ("fastest fifty on ODI debut"), is the web being newer than
the library: it reports a 24-ball fifty from March 2025 that beats the golden answer's 26-ball record. Counted
right, the web run ties the final at 0.974; it never beats it.

### The question nothing fixed: when more evidence hurts

trap-05: "What is the highest individual Test score by an Indian batter against England?" (Karun Nair, 303*, 2016).
Every build from Adaptive #4 on, including this one, answered "Virat Kohli, 235". Across every saved run:

| Builds | Passages read | trap-05 |
|---|---|---|
| Contextual #2, Fusion #3, two CRAG #5 variants | 5 | ✓ Nair 303* |
| Adaptive #4, CRAG #5 final, Self-RAG #6, GraphRAG #7, every Agentic run | 7–9 | ✗ Kohli 235 |

The simple builds read 5 passages, mostly from the 2016–17 series article, where the 303* sits on a scorecard. The
richer builds search more, and their extra passages include the Wankhede Stadium article, which says Kohli's 235
was "the highest score by an Indian against England", a sentence written a week before Nair passed it and never
updated. Once that sentence is in front of gpt-4o-mini, it trusts the written claim over the bigger number. In this
build the agent even doubted the walk ("unclear which is the highest"), then said "enough" once IRCoT brought the
stale sentence, so the web rung never fired. I did not fix it: every fix I could see would have been tuned to this
one question.

## Test: 20 fresh questions, run once

I wrote 20 questions from the shared library, by reading text, not by running any system. `agentic/agent_golden.py`
checked them for free (no search, no LLM): every answer is in the corpus word for word, each two-step question
needs two different articles, each decoy really is in the library, and each gap answer is only in the fallback
library.

| Kind | n | What it tests |
|---|---|---|
| chain | 5 | the walk: hop 1 on a match record (player of the match, venue), hop 2 in another article |
| pair | 3 | two facts from two articles |
| twin | 2 | a similar match or innings holds a tempting wrong answer |
| alike | 3 | record questions with "highest/lowest at X" decoys; the answer is **only in the record pages** |
| solo, gap, sure, none | 2, 2, 1, 2 | a lookup, the fallback library, the router's memory route, a refusal |

Only #8 has the record pages, so the fair comparison is on the other 17. `agentic/agent_test.py` ran all three
setups once, saved `agentic/results/test_once.json` and refuses to run again.

| Test 20 | GraphRAG #7 FINAL | **Agentic #8 FINAL** | + web |
|---|---|---|---|
| right, 1 vote / 3 votes | 0.825 / 0.825 | **0.975 / 0.975** | 0.925 / 0.925 |
| main 17 (fair) | 0.912 | **0.971** | 0.971 |
| record questions | 1/3 | **3/3** | 2/3 |
| multi (13) | 12.5 | 12.5 | 12.5 |
| ladder paths | — | walk 8, walk → IRCoT 5 | + web on 2 multi, 6 single |
| $ per question / median | $0.0043 / 17.0 s | **$0.0029 / 9.0 s** | $0.0030 / 9.8 s |

| Question | GraphRAG → Agentic | Why |
|---|---|---|
| twin-31: Root, Lord's 2021, first innings | 0 → 1 | Same passages, same CRAG grades, CRAG's answer right (180*). In GraphRAG, Self-RAG's critic preferred another candidate: "64, at Trent Bridge" |
| alike-32: lowest Test total | 0 → 1 | GraphRAG took the decoy, India's 36 ("their lowest team total in Tests"). The record page says New Zealand, 26 |
| alike-33: highest T20I score | 0 → 1 | GraphRAG: David Warner's 89. The record page: Aaron Finch, 172 |
| alike-33 with web | 1 → 0 | The agent searched "highest individual score in T20 International cricket **2023**", adding a year the question never asked for. 2023 season stats came first in the pool and the answer became "I don't know" |

pair-33 scored 0.5 in all three setups: the answer gives Gill's birthplace as "Chak Jaimal Singh Wala, Fazilka
district, Punjab", which is right and more precise than the golden "Fazilka, Punjab". I left the score as graded.

**The two sets agree.** GraphRAG #7 scored 0.812 on Self-RAG's test and 0.825 on this one, so the new questions are
not easier for it. On tune the final system was 0.974; on unseen questions 0.975. Unlike GraphRAG's walk, this
build's choices held up on new questions, largely because its gains came from data and from removing a step, not
from fitting a route to 8 questions.

## What the agent itself contributed

Taken apart, the agent's ladder was neither the main gain nor a loss:
- **On multi it gained half a question on tune (chain-06) and tied on test.** It recovers from a wrong walk only
  when it notices the walk fell short, and the walk was right on most questions.
- **It climbs to IRCoT on about half the multi questions** (22 of 39 on tune, 5 of 13 on test), so a multi question
  takes longer than GraphRAG's on tune (median 18.1 s vs 12.0 s). On test, without Self-RAG's layer, it was faster
  (9.5 s vs 15.9 s).
- **Its decisions are cheap and replayable:** one JSON call per decision through the LLM cache, so any saved run
  replays for $0.

The bigger levers were around it: what the library holds and what judges the final answer.

## Run it

From `agentic_rag_in_depth/`, next to working `graph_rag_in_depth/`, `self_rag_in_depth/`,
`corrective_rag_in_depth/`, `adaptive_rag_in_depth/`, `rag_fusion_in_depth/`, `contextual_rag_in_depth/` and
`eval_harness/`. Qdrant and Neo4j run in Docker (`docker start qdrant kg-neo4j`). Source the OpenAI key (and
`SERPER_API_KEY` for anything with web) for whatever is not yet cached.

```bash
OPENAI_API_KEY=sk-dummy uv run agentic/replay_graph.py   # Phase 1: GraphRAG's tune score, from the cache
OPENAI_API_KEY=sk-dummy uv run agentic/agent_dry.py      # Phase 2: free check of cues, tools and prompt
uv run agentic/agent_smoke.py                            # Phase 2: the ladder on 5 questions, both modes
uv run agentic/agent_webcheck.py                         # Phase 3: the web rung end to end on one question
uv run agentic/agent_tune.py                             # Phase 3: 78 tune questions, without and with web
uv run agentic/agent_noselfrag.py                        # Phase 3: offline check, answers without Self-RAG's critic
uv run agentic/corpus_gap.py fetch                       # Phase 3: the record pages (fetch / build / notes / index)
uv run agentic/corpus_gap.py check C_records_noselfrag   # Phase 3: tune runs A-D (C is the final setup)
uv run python -I -c "import sys; sys.path.insert(0,'agentic'); import agent_golden"   # Phase 4: check the test set
uv run python -I -c "import sys; sys.path.insert(0,'agentic'); import agent_test; agent_test.main()"   # runs once
```

## Spend

| Item | Cost |
|---|---|
| Phase 2 smokes (three agents) | $0.06 |
| Phase 3 tune runs, web checks and offline checks | $0.41 |
| Record pages' notes | $0.07 |
| Test (three setups) | $0.13 |
| **Total** | **≈ $0.67 of gpt-4o-mini**, about 60 Serper searches |

## Lessons

- **A small model gates better than it plans.** gpt-4o-mini picking among six tools gave two different strategies
  for two prompt layouts, both worse than no agent. The same model answering "is this enough?" over evidence it
  could read, with a fixed order of tools behind it, worked on every smoke question.
- **Corpus coverage comes before architecture.** No retrieval method finds a page that is not in the library.
  Hybrid search, notes, fusion, routing, grading, checking, a graph and an agent could not answer "highest score in
  the IPL"; six record pages did, and broke nothing.
- **More evidence can hurt.** Builds that read 5 passages got trap-05 right; every build that read more got it wrong,
  because more searching also finds a confident, stale sentence, and the model trusts claims over numbers.
- **A weak judge costs more than it catches.** Self-RAG's critic matched CRAG's answer almost everywhere, rejected
  right web answers, and on the test set picked a wrong candidate over a right one.
- **Don't let the LLM's beliefs into the grader.** The agent's "known facts" made CRAG accept wrong passages. Graph
  links can guide grading because they come from records, not from the model.
- **The web is fresher than a pinned library, and that cuts both ways.** It was right where the golden answer was
  out of date (trap-02), and wrong when the agent added its own year to the query (alike-33). It never beat the
  library once the library had the right pages.
- **A safety net is only as good as its trigger.** A look-alike graded 10 switches off CRAG's fallback; a stale
  claim makes the agent say "enough". In this build the misses came from nets that never fired.

## When I'd reach for it

An agent on top of a strong pipeline pays when the first retrieval step is cheap and usually right, and a slower
step can recover when it isn't: the ladder runs the expensive step only on the half of the questions that need it.
With a small model I'd keep the agent's job to a yes/no on evidence and fix the order of tools in code. Before any
of it, I'd check that the library holds the pages its users' simplest questions need.

## Where #8 sits in the series

Each build was measured against the one before it. On Self-RAG's 78 tune questions, the same ruler for the last
four builds:

| Build | Adds | Tune (78) | Test |
|---|---|---|---|
| CRAG #5 stack (Contextual + Fusion + Adaptive + CRAG 3f) | the base the last three builds sit on | 0.910 | — |
| Self-RAG #6 | candidates + critic + "are you sure?" | 0.897 | 0.875 (Self-RAG test 24) |
| GraphRAG #7 | graph walk on multi, linked grading, IRCoT fallback | 0.942 | 0.812 (Self-RAG test 24) · 0.825 (Agentic test 20) |
| **Agentic #8** | ladder agent, record pages, critic removed | **0.974** | **0.975** (Agentic test 20) |

What survives into the final system from every build: Contextual #2's chunk notes, Fusion #3's f6a search, Adaptive
#4's router and IRCoT, Corrective #5's grader and fallback library, Self-RAG #6's "are you sure?", GraphRAG #7's walk
and linked grading, and #8's ladder and record pages.

## Files

| File | Purpose |
|---|---|
| `agentic/agentic_bridge.py` | Puts GraphRAG #7's code (and through it every earlier build's) on the import path; copies its LLM cache |
| `agentic/replay_graph.py` | Phase 1: GraphRAG's tune score through the bridge, from the cache |
| `agentic/agent_web.py` | Serper web search with a disk cache (`data/cache/web.json`) |
| `agentic/agent_tools.py` | The three tools: graph walk, IRCoT, web search |
| `agentic/agent_loop.py` | The ladder, the "enough?" prompt, the pool, linked grading; `install()` sets web, Self-RAG and the single route |
| `agentic/agent_dry.py` | Phase 2: free check of the walk's cues, the tools' wiring and the prompt |
| `agentic/agent_smoke.py` | Phase 2: 5 questions, both modes (`phase2_smoke_v1/_v2.json` are the free-choice agents) |
| `agentic/agent_webcheck.py` | Phase 3: the web rung end to end, forced |
| `agentic/agent_tune.py` | Phase 3: tune runs without and with web |
| `agentic/agent_noselfrag.py` | Phase 3: offline check of answers without Self-RAG's critic |
| `agentic/corpus_gap.py` | Phase 3: the record pages (fetch, build, notes, index, `install()`) and tune runs A–D |
| `agentic/agent_golden.py` | Phase 4: builds and checks the 20 test questions, free |
| `agentic/agent_test.py` | Phase 4: the one test run of all three setups |
| `labels/new_test.jsonl`, `labels/decoys.json` | The 20 test questions and their decoys |
| `phase3_log.md`, `corpus_gap_log.md` | Phase 2–3 problems and fixes; the corpus-gap experiment |
| `agentic/results/*.json` | Every phase's numbers, per question (`test_once.json` is the test) |
