"""The web tool: Google results through Serper (google.serper.dev), cached on disk by query.

Web results change from day to day, so every reply is kept in data/cache/web.json the first time a query is sent:
a replay reads the cache and makes no request, and two runs of the same question see the same results. Only the
query text goes to Serper. The key is read from SERPER_API_KEY (RAGS/rag_sprints/.env) and never printed.

Each result becomes a passage the rest of the system can read like a library chunk: the answer box (when Google
shows one) and the top organic results, as "Web › <title>" + the snippet + its link.
"""

import hashlib
import json
import os
from pathlib import Path

import httpx

import agentic_bridge as ab

CACHE = ab.ROOT / "data" / "cache" / "web.json"
URL = "https://google.serper.dev/search"
N_RESULTS = 5
USAGE = {"requests": 0, "cached": 0}  # per run, for the smoke and tune reports
_cache: dict | None = None


def _load() -> dict:
    global _cache
    if _cache is None:
        _cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    return _cache


def key(query: str) -> str:
    return hashlib.sha1(query.strip().lower().encode()).hexdigest()[:16]


def search(query: str) -> list[dict]:
    """[{"title", "link", "snippet"}], answer box first. Cached by the lower-cased query."""
    cache, k = _load(), key(query)
    if k in cache:
        USAGE["cached"] += 1
        return cache[k]["results"]
    api_key = os.environ.get("SERPER_API_KEY")
    if not api_key:
        raise RuntimeError("SERPER_API_KEY is not set (source RAGS/rag_sprints/.env)")
    resp = httpx.post(URL, headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
                      json={"q": query, "num": N_RESULTS}, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    results = []
    box = data.get("answerBox") or {}
    if box.get("answer") or box.get("snippet"):
        results.append({"title": box.get("title", "Answer box"), "link": box.get("link", ""),
                        "snippet": box.get("answer") or box.get("snippet")})
    for r in data.get("organic", [])[:N_RESULTS]:
        results.append({"title": r.get("title", ""), "link": r.get("link", ""), "snippet": r.get("snippet", "")})
    cache[k] = {"query": query, "results": results}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(cache, indent=1, ensure_ascii=False))
    USAGE["requests"] += 1
    return results
