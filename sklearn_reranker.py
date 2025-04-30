import asyncio
from typing import List, Optional

import aiohttp
import numpy as np
from aiohttp import TCPConnector
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from starlette.middleware.cors import CORSMiddleware
import ssl
import certifi

HN_TOP_STORIES_URL = "https://hacker-news.firebaseio.com/v0/topstories.json"
HN_ITEM_URL = "https://hacker-news.firebaseio.com/v0/item/{}.json"

app = FastAPI(title="HN Relevance Reranker", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # allow requests from any origin
    allow_credentials=True,
    allow_methods=["*"],            # allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],            # allow all headers
)


class BioRequest(BaseModel):
    """Input schema – user bio as free text."""

    bio: str


class HNItem(BaseModel):
    """Output schema for each Hacker News story."""

    id: int
    title: str
    url: Optional[str] = None
    score: Optional[int] = None
    similarity: float


# ---------------------------------------------------------------------------
# Helpers to fetch Hacker News data asynchronously
# ---------------------------------------------------------------------------


async def _fetch_json(session: aiohttp.ClientSession, url: str) -> dict:
    """Fetch a JSON URL or raise HTTPException on non‑200."""

    async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as r:
        if r.status != 200:
            raise HTTPException(
                status_code=502, detail=f"Error fetching {url} → {r.status}"
            )
        return await r.json()


async def _fetch_story(session: aiohttp.ClientSession, sid: int) -> dict:
    """Fetch a single HN story dict; returns empty dict for deleted items."""

    try:
        return await _fetch_json(session, HN_ITEM_URL.format(sid))
    except Exception:
        return {}


async def _get_top_story_items(limit: int = 500) -> List[dict]:
    """Return up to *limit* HN top‑stories dicts (front page ≈30 but API gives 500)."""
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    connector = TCPConnector(ssl=ssl_context)

    async with aiohttp.ClientSession(connector=connector) as session:
        ids: List[int] = await _fetch_json(session, HN_TOP_STORIES_URL)
        ids = ids[:limit]

        # polite concurrency: cap at 50 simultaneous requests
        sem = asyncio.Semaphore(50)

        async def _bounded_fetch(i: int):
            async with sem:
                return await _fetch_story(session, i)

        items = await asyncio.gather(*(_bounded_fetch(i) for i in ids))
        return [itm for itm in items if itm]


# ---------------------------------------------------------------------------
# Text relevance – TF‑IDF + cosine similarity (fast, no 3rd‑party services)
# ---------------------------------------------------------------------------


def _rerank_stories(user_bio: str, stories: List[dict]) -> List[HNItem]:
    """Return *stories* sorted by relevance to *user_bio*."""

    documents = [user_bio] + [s.get("title", "") for s in stories]
    tfidf = TfidfVectorizer(stop_words="english")
    matrix = tfidf.fit_transform(documents)
    sims: np.ndarray = cosine_similarity(matrix[0:1], matrix[1:]).flatten()

    ranked_indices = np.argsort(-sims)  # descending order
    ranked_items: List[HNItem] = []

    for idx in ranked_indices:
        s = stories[idx]
        ranked_items.append(
            HNItem(
                id=s["id"],
                title=s.get("title", ""),
                url=s.get("url"),
                score=s.get("score"),
                similarity=float(sims[idx]),
            )
        )

    return ranked_items


@app.post("/rerank", response_model=List[HNItem])
async def rerank(req: BioRequest):
    """Return top 500 HN stories ranked by relevance to *req.bio*."""

    stories = await _get_top_story_items()
    if not stories:
        raise HTTPException(status_code=502, detail="Failed to fetch Hacker News stories")

    return _rerank_stories(req.bio, stories)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
