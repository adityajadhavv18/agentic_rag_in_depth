"""Reuse GraphRAG #7's final system as-is (and through it Self-RAG #6, Corrective #5, Adaptive #4, RAG Fusion #3 and
Contextual RAG #2): router, f6a search, graph walk on Neo4j, CRAG 3f with linked grading and its fallback library,
IRCoT fallback, and the Self-RAG layer.

GraphRAG's grag/ folder goes on sys.path; its grag_bridge adds Self-RAG's, which adds the rest. Every module there is
imported by bare name, so ours must not reuse any of their names:
    Contextual #2: chunks, ingest, retrieve, hyde, llm_rerank, notes, ladder, answer
    Fusion #3:     contextual, rewrite, rag_fusion, diagnose, fusion_ladder, fusion_answer
    Adaptive #4:   bridge, bar, golden_check, llm_cache, routes, route_eval, loop_knobs, router, router_eval,
                   end_to_end, stacked, answer_tuning, test_run
    Corrective #5: crag_bridge, crag, crag_eval, grader, grader_eval, baseline, chunk_labels, crag_golden_check,
                   fallback_index, laya_eval, model_check, test_once, test_relabel, compare_archs
    Self-RAG #6:   selfrag_bridge, base_tune, replay_3f, self_rag, selfrag_golden, selfrag_tune, smoke
    GraphRAG #7:   grag_bridge, extract_code, graph_route, graph_test, graph_tune, hop_coverage, linked_check,
                   load_neo4j, records_index, records_notes, records_route, records_dry, records_smoke,
                   replay_selfrag, walk, walk_dry

GraphRAG's LLM cache is copied once into this build's data/cache/llm.json and llm_cache is pointed at the copy:
replays are free and new calls land here. The system the agent plugs into is GraphRAG FINAL (tune 0.942).
"""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAGS = ROOT.parent
GRAG = RAGS / "graph_rag_in_depth" / "grag"
sys.path.append(str(GRAG))

import grag_bridge  # noqa: E402  (GraphRAG's bridge: adds Self-RAG, Corrective, Adaptive, Fusion, Contextual)
import graph_route  # noqa: E402
import llm_cache  # noqa: E402

CACHE = ROOT / "data" / "cache" / "llm.json"
if not CACHE.exists():
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(grag_bridge.CACHE, CACHE)
llm_cache.CACHE = CACHE  # set before any call: llm_cache loads its file lazily

SELFRAG_CONFIG = grag_bridge.SELFRAG_CONFIG  # {"sure": True, "retries": 1}
FINAL = graph_route.FINAL  # {"linked": True, "add": False, "fallback": True}
GRAG_RESULTS = GRAG / "results"
RESULTS = ROOT / "agentic" / "results"
