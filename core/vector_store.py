"""
In-memory vector store replacing ChromaDB.
Uses numpy cosine similarity — no C extension dependencies.
"""
import hashlib
import logging
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from config.settings import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

_embedder = None
_store: List[Dict] = []  # list of {id, doc, metadata, embedding}

def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    return _embedder

def get_collection(name="research_papers"):
    return _store

def ingest_papers(papers, collection_name="research_papers"):
    embedder = _get_embedder()
    new_count = 0
    existing_ids = {p["id"] for p in _store}
    for p in papers:
        doc = f"{p.get('title','')}. {p.get('abstract','')}"
        doc_id = hashlib.md5(doc.encode()).hexdigest()
        if doc_id in existing_ids:
            continue
        emb = embedder.encode([doc])[0]
        _store.append({
            "id": doc_id,
            "doc": doc,
            "embedding": emb,
            "metadata": {
                "title": p.get("title",""),
                "authors": ", ".join(p.get("authors",[])),
                "year": str(p.get("year","")),
                "url": p.get("url",""),
                "source": p.get("source",""),
                "paper_id": p.get("paper_id",""),
                "citation_count": str(p.get("citation_count",0) or 0),
                "influential_citation_count": str(p.get("influential_citation_count",0) or 0),
                "journal": p.get("journal",""),
            }
        })
        existing_ids.add(doc_id)
        new_count += 1
    return new_count

def similarity_search(query, n_results=10, collection_name="research_papers"):
    if not _store:
        return []
    embedder = _get_embedder()
    q_emb = embedder.encode([query])[0]
    results = []
    for item in _store:
        emb = item["embedding"]
        sim = float(np.dot(q_emb, emb) / (np.linalg.norm(q_emb) * np.linalg.norm(emb) + 1e-9))
        results.append({**item["metadata"], "snippet": item["doc"][:300], "similarity": round(sim, 4)})
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:n_results]

def reset_collection(collection_name="research_papers"):
    global _store
    _store = []
    logger.info("In-memory store cleared.")

def collection_stats(collection_name="research_papers"):
    return {"collection": collection_name, "total_papers": len(_store)}