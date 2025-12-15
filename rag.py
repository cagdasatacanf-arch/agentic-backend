from typing import List, Tuple, Dict

import httpx
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from app.config import settings

OPENAI_API_KEY = settings.openai_api_key
CHAT_MODEL = settings.openai_chat_model
EMBED_MODEL = settings.openai_embedding_model
QDRANT_URL = settings.vector_db_url
COLLECTION = "docs"

client = QdrantClient(url=QDRANT_URL)


async def ensure_collection():
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION not in existing:
        client.recreate_collection(
            COLLECTION,
            vectors=VectorParams(size=3072, distance=Distance.COSINE),
        )


async def embed(texts: List[str]) -> List[List[float]]:
    async with httpx.AsyncClient(timeout=30.0) as http:
        resp = await http.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={"model": EMBED_MODEL, "input": texts},
        )
    resp.raise_for_status()
    data = resp.json()["data"]
    return [d["embedding"] for d in data]


async def search_docs(query: str, top_k: int = 5) -> List[Dict]:
    await ensure_collection()
    vec = (await embed([query]))[0]
    res = client.search(collection_name=COLLECTION, query_vector=vec, limit=top_k)
    docs: List[Dict] = []
    for r in res:
        docs.append(
            {
                "id": str(r.id),
                "text": r.payload.get("text", ""),
                "score": r.score,
                "metadata": r.payload.get("metadata", {}),
            }
        )
    return docs


async def get_agent_answer(
    question: str,
    top_k: int = 5,
) -> Tuple[str, List[Dict], Dict]:
    docs = await search_docs(question, top_k=top_k)
    context = "\n\n".join(d["text"] for d in docs)

    prompt = (
        "You are an AI agent answering questions using the context below.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\n\n"
        "Answer clearly and say when the context is insufficient."
    )

    async with httpx.AsyncClient(timeout=60.0) as http:
        resp = await http.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={
                "model": CHAT_MODEL,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
    resp.raise_for_status()
    data = resp.json()
    answer = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})

    return answer, docs, usage
