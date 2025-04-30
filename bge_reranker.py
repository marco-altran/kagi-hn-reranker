import asyncio
import os
import ssl
from typing import List, Dict, Any

import aiohttp
import certifi
from aiohttp import TCPConnector
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ----------------------- model load (BAAI/bge-reranker-large) -----------------
MODEL_NAME = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-large")
device     = torch.device("mps" if torch.backends.mps.is_built() else "cpu")

tokenizer   = AutoTokenizer.from_pretrained(MODEL_NAME)
model       = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
# Reduce the precision for faster inference while maintaining quality
model.half().to(device).eval()

# ---------------------------- FastAPI app -------------------------------------
app = FastAPI(title="HN Re-ranker", version="1.0")

class Bio(BaseModel):
    bio: str

HN_TOP_URL   = "https://hacker-news.firebaseio.com/v0/topstories.json"
HN_ITEM_URL  = "https://hacker-news.firebaseio.com/v0/item/{id}.json"

# ------------------------ helper: fetch HN stories ----------------------------
async def fetch_json(session: aiohttp.ClientSession, url: str) -> Dict[str, Any]:
    async with session.get(url, timeout=10) as resp:
        if resp.status != 200:
            raise HTTPException(status_code=502, detail=f"HN returned {resp.status}")
        return await resp.json()

async def get_top_items(n: int = 500) -> List[Dict[str, Any]]:
    """
    Fetch metadata for the top *n* Hacker News stories concurrently.
    """
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    connector = TCPConnector(ssl=ssl_context)

    async with aiohttp.ClientSession(connector=connector) as session:
        ids = await fetch_json(session, HN_TOP_URL)
        ids = ids[:n]                                              # cap at 500
        sem = asyncio.Semaphore(50)                                # limit concurrency

        async def bound_fetch(i):
            async with sem:
                return await fetch_json(session, HN_ITEM_URL.format(id=i))

        tasks = [asyncio.create_task(bound_fetch(i)) for i in ids]
        items = await asyncio.gather(*tasks)
        return [it for it in items if it]                          # filter None

# ------------------------ helper: rerank with BGE -----------------------------
@torch.inference_mode()
def rerank(bio: str, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Only rank the title for speed and easier evaluation
    texts = [f"{it.get('title','')}" for it in items]
    batch_size = 64

    scores_list = []

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            # encode
            encodings = tokenizer(
                [bio] * len(batch_texts),
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            # move to device
            input_ids = encodings["input_ids"].to(device)
            attention_mask = encodings["attention_mask"].to(device)

            # model forward
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # logits: shape (batch_size, 1) → squeeze to (batch_size,)
            batch_scores = outputs.logits.squeeze(-1)

            scores_list.append(batch_scores)

    # concatenate all batch scores into a single tensor on CPU
    scores = torch.cat(scores_list, dim=0)

    # sort and build the ranked list
    order = torch.argsort(scores, descending=True).tolist()
    ranked = []
    for idx in order:
        item = items[idx]
        ranked.append({
            "id":    item["id"],
            "title": item.get("title", ""),
            "url":   item.get("url", f"https://news.ycombinator.com/item?id={item['id']}"),
            "similarity": float(scores[idx]),
        })
    return ranked

# -------------------------------- endpoint ------------------------------------
@app.post("/rerank", summary="Return 500 HN stories re-ranked by relevance")
async def rank(bio: Bio):
    items  = await get_top_items()
    result = rerank(bio.bio, items)
    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8081)