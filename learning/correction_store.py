"""
Correction Store — Pinecone + Jina AI Embeddings.

Stores doctor correction patterns as vectors in Pinecone cloud.
- Jina AI: pure HTTP, no SDK, works on Railway/Vercel/anywhere, 1M tokens/month free
- Pinecone: serverless cloud vector DB, corrections persist forever
- 1024-dim vectors (jina-embeddings-v3)
- Cold-start solved: DB persists across all restarts and deployments
"""
import os
import json
import uuid
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

from project_paths import PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX   = os.getenv("PINECONE_INDEX", "discharge-corrections")
JINA_API_KEY     = os.getenv("JINA_API_KEY")
EMBED_DIM        = 1024   # jina-embeddings-v3 default

_pinecone_index  = None


def _get_index():
    global _pinecone_index
    if _pinecone_index is not None:
        return _pinecone_index
    from pinecone import Pinecone, ServerlessSpec
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing = [i.name for i in pc.list_indexes()]
    if PINECONE_INDEX not in existing:
        print(f"Creating Pinecone index '{PINECONE_INDEX}' ({EMBED_DIM} dims)...")
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        while not pc.describe_index(PINECONE_INDEX).status["ready"]:
            time.sleep(1)
        print("Pinecone index ready.")
    _pinecone_index = pc.Index(PINECONE_INDEX)
    return _pinecone_index


def embed(texts: list) -> list:
    """
    Embed texts using Jina AI (jina-embeddings-v3, 1024-dim).
    Uses requests library — works on Railway, Vercel, anywhere. No local model.
    """
    for attempt in range(3):
        try:
            resp = requests.post(
                "https://api.jina.ai/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {JINA_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "jina-embeddings-v3",
                    "input": texts,
                    "task": "text-matching",
                    "dimensions": EMBED_DIM
                },
                timeout=30
            )
            if resp.status_code == 429 and attempt < 2:
                wait = 2 ** (attempt + 1)
                print(f"  Jina rate limit — waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return [item["embedding"] for item in resp.json()["data"]]
        except requests.HTTPError as e:
            if attempt == 2:
                raise RuntimeError(f"Jina embed error {e.response.status_code}: {e.response.text[:200]}")
            time.sleep(2)
        except Exception as e:
            if attempt == 2:
                raise RuntimeError(f"Jina embed failed: {e}")
            time.sleep(2)

    raise RuntimeError("Jina embed: all retries exhausted")


def store_correction(correction: dict) -> str:
    """
    Store a doctor correction pattern in Pinecone as a vector.
    Returns the stored vector ID.
    correction keys: section, agent_wrote, doctor_changed_to, pattern,
                     patient_id, iteration, reward_delta
    """
    idx = _get_index()
    text_to_embed = (
        f"Section: {correction.get('section', '')}. "
        f"Pattern: {correction.get('pattern', '')}. "
        f"Example: {str(correction.get('agent_wrote', ''))[:200]}"
    )
    embedding = embed([text_to_embed])[0]
    vector_id = str(uuid.uuid4())
    idx.upsert(vectors=[{
        "id": vector_id,
        "values": embedding,
        "metadata": {
            "section":           correction.get("section", ""),
            "agent_wrote":       str(correction.get("agent_wrote", ""))[:500],
            "doctor_changed_to": str(correction.get("doctor_changed_to", ""))[:500],
            "pattern":           str(correction.get("pattern", ""))[:500],
            "patient_id":        correction.get("patient_id", ""),
            "iteration":         int(correction.get("iteration", 0)),
            "reward_delta":      float(correction.get("reward_delta", 0.0)),
        }
    }])
    return vector_id


def store_corrections_batch(corrections: list) -> list:
    """Store a batch of corrections. Embeds one at a time to stay under rate limits."""
    ids = []
    for c in corrections:
        ids.append(store_correction(c))
        time.sleep(0.5)   # small pause — Jina free tier is generous but be polite
    return ids


def query_relevant_corrections(section: str, context: str, top_k: int = 5) -> list:
    """
    Retrieve most relevant past corrections from Pinecone for the given context.
    Used to inject learned patterns into the agent before each run.
    """
    idx = _get_index()
    query_text = f"Section: {section}. Context: {context[:300]}"
    query_embedding = embed([query_text])[0]
    results = idx.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    corrections = []
    for match in results.matches:
        if match.score > 0.3:
            corrections.append({"score": round(match.score, 3), **match.metadata})
    return corrections


def get_all_corrections(limit: int = 100) -> list:
    """Fetch stored corrections (uses broad query since Pinecone free tier has no list-all)."""
    idx = _get_index()
    dummy_vec = [0.0] * EMBED_DIM
    results = idx.query(vector=dummy_vec, top_k=limit, include_metadata=True)
    return [{"id": m.id, "score": round(m.score, 3), **m.metadata} for m in results.matches]


def get_correction_count() -> int:
    """Return total corrections stored in Pinecone."""
    try:
        idx = _get_index()
        return idx.describe_index_stats().total_vector_count
    except Exception:
        return 0


def format_corrections_for_prompt(corrections: list) -> str:
    """Format retrieved corrections for injection into the agent system prompt."""
    if not corrections:
        return ""
    lines = [
        "\n=== LEARNED CORRECTIONS FROM PAST DOCTOR REVIEWS ===",
        "Apply these patterns — they reflect consistent clinician preferences:\n"
    ]
    for i, c in enumerate(corrections, 1):
        lines.append(f"{i}. [{c.get('section','').upper()}] {c.get('pattern','')}")
        if c.get("agent_wrote") and c.get("doctor_changed_to"):
            lines.append(f"   Before: {c['agent_wrote'][:120]}")
            lines.append(f"   After:  {c['doctor_changed_to'][:120]}")
        lines.append("")
    lines.append("=== END OF LEARNED CORRECTIONS ===\n")
    return "\n".join(lines)
