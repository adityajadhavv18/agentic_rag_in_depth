# Corpus gap log: simple record questions the architectures could not answer

## The problem

"What is the highest individual score in an IPL match, and who made it?" is one of the simplest questions anyone
asks about the IPL. The answer is Chris Gayle's 175 not out (2013). My system got it wrong in every build measured on
the 78-question tune set: Self-RAG #6, GraphRAG #7 and the Agentic #8 ladder. Two similar questions failed the same
way:

| Question | Gold | What the system said |
|---|---|---|
| alike-05: highest individual score in an IPL match | Chris Gayle, 175* (2013) | "Rajat Patidar, 112" (the highest score **at Eden Gardens**) |
| alike-04: best bowling figures in IPL history | Alzarri Joseph, 6/12 (2019) | "I don't know" |
| trap-05: highest Test score by an Indian against England | Karun Nair, 303* (2016) | "Virat Kohli, 235" |

## What I found

**The library has no record pages.** It holds 149 Wikipedia articles: every IPL season and final from 2008, the
Ashes, Border–Gavaskar and India–England series, World Cups, Champions Trophies, WTC finals, grounds and 30 player
biographies. Not one of them is a records or statistics page.

**The answers appear only in passing, in other words than the question's.**
- Gayle's record is one sentence of the 2013 IPL season article: "Gayle's score of 175 not out, **the highest in
  Twenty20 cricket**". The question says "IPL"; the sentence says "Twenty20 cricket".
- Alzarri Joseph's 6/12 is in a 2019 season passage that search ranks 46th, below the cut at 20.
- Nair's 303* is a scorecard line. The only sentence that says "highest score by an Indian against England" is about
  Kohli's 235, written a week before Nair passed it, and never updated.

**Look-alikes win, and they fool the safety nets.** Dozens of passages say "highest score at this ground" or "highest
this season". CRAG's grader scored a look-alike 10 for alike-05, said CORRECT, and so neither its fallback library nor
a last-resort web search ever ran.

**The web had the answer at once.** A Google search through Serper returned "Chris Gayle holds the record for highest
individual score in IPL with 175 not out in 2013". Made compulsory on every single-route question, web search fixed
alike-04 and alike-05 but broke three questions the library already answered (a true fact about another innings, a
snippet too short to answer, and passages that switched off CRAG's fallback).

## The lesson

No retrieval architecture can find a page that is not in the corpus. Across the series I added hybrid search,
contextual notes, query fusion, adaptive routing, corrective grading, self-checking, a knowledge graph and an agent
that escalates. Each fixed real failures, and none of them could answer the simplest record question, because the
library never held the record book. Corpus coverage is a precondition of RAG, not a detail: before tuning retrieval,
check that the pages your users' simplest questions need are in the library.

## The check (Agentic #8 only)

I add the record pages to the library for this build only, in a separate Qdrant collection, so builds #1 to #7 keep
their corpus and their numbers. One records page per format and per tournament the library covers:

- List of Indian Premier League records and statistics
- List of Test cricket records
- List of One Day International cricket records
- List of Twenty20 International records
- List of Cricket World Cup records
- List of ICC Men's T20 World Cup records
- List of ICC Champions Trophy records

They are chosen by the library's scope, not by the failing questions: no page is added for a single question (there
is no India-v-England records page, so trap-05 is not expected to change). "List of ICC Champions Trophy records"
does not exist on Wikipedia (it redirects to the tournament's main article), so six pages went in. Each page is
fetched at a pinned revision, cleaned and chunked exactly like the library (300-token windows, title › heading
labels), given the same LLM note, and indexed beside the 149 articles: 406 new chunks next to 4,650, in a separate
collection (the library's own collection is unchanged).

Then the 78 tune questions run through the Agentic #8 system without web, and I report every question that changes,
gains and breaks.

### Results

Before any question ran, a free check of the search alone: with the record pages in, the IPL records page ranks
1st and 2nd for both IPL record questions; without them it is in no list's top 20.

| Setup (no web) | Tune (78), 1 vote / 3 votes | Changed | $/question | Median time |
|---|---|---|---|---|
| Agent, old library, Self-RAG on (Phase 3) | 0.949 / 0.949 | — | $0.0049 | 12.4 s |
| Agent, old library, Self-RAG off | 0.949 / 0.949 | none | $0.0029 | 8.9 s |
| **Agent + record pages, Self-RAG on** | **0.974 / 0.974** | alike-04 0 → 1, alike-05 0 → 1 | $0.0048 | 14.9 s |
| **Agent + record pages, Self-RAG off** | **0.974 / 0.974** | alike-04 0 → 1, alike-05 0 → 1 | $0.0029 | 9.3 s |

- **Both record questions fixed, nothing broken.** "The highest individual score in an IPL match is 175*, made by
  Chris Gayle" and "Alzarri Joseph… 6/12", each cited from the IPL records page.
- CRAG kept record passages on 9 questions (alike-03 to alike-06, trap-01, trap-02, trap-05, trap-06, fame-03), and
  none of them got worse: the extra pages did not mislead questions they had no answer for.
- **trap-05 stays wrong**, as expected: no added page names the highest Test score by an Indian against England.
- The last miss, none-11, is a question that should be refused ("I don't know"), and a record page cannot fix that.

**With the record pages in, web search added nothing.** Run D (records on, Self-RAG off, web as a last resort on
both routes: the agent asks "enough?" and only then searches Google) scored 0.962. The agent sent 19 of 30 single
questions and 9 multi questions to the web, gained none, and changed one answer: trap-02 ("fastest fifty on ODI
debut"). There the web was more up to date than the library. Its results say Muhammad Abbas made a 24-ball fifty
on debut in March 2025, breaking Krunal Pandya's 26-ball record, while the pinned library and the golden answer
still say Pandya. No question that should be refused was broken.

Six record pages fixed what eight architectures and a web search had not fixed safely, at no cost to any other
question. The cheapest, most accurate fix for a simple question the system can't answer is often the data, not the
architecture.
