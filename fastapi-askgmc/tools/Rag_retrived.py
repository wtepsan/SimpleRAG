
import os
import json
import pickle
import math
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import faiss

from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer


# =========================
# Helpers
# =========================
def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def chunk_text(text: str, max_length: int = 0, overlap: int = 0) -> List[str]:
    """Character-based chunking with optional overlap."""
    if not isinstance(text, str) or not text.strip():
        return []
    if max_length <= 0:
        return [text.strip()]
    chunks = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + max_length, n)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def _encode(documents: List[str], model_name: str) -> np.ndarray:
    if not documents:
        raise ValueError("No documents to encode.")
    model = SentenceTransformer(model_name)
    vecs = model.encode(documents, show_progress_bar=True, normalize_embeddings=True)
    return np.asarray(vecs, dtype="float32")


def _build_faiss(vectors: np.ndarray) -> faiss.Index:
    dim = int(vectors.shape[1])
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index


def _build_tfidf(documents: List[str]) -> Tuple[TfidfVectorizer, sparse.csr_matrix]:
    vec = TfidfVectorizer()
    mat = vec.fit_transform(documents)
    return vec, mat


def _save_artifacts(
    output_dir: str,
    prefix: str,
    info_dict: Dict[str, Any],
    vectors: np.ndarray,
    faiss_index: faiss.Index,
    tfidf_vectorizer: TfidfVectorizer,
    tfidf_matrix: sparse.csr_matrix,
):
    _ensure_dir(output_dir)

    info_path = os.path.join(output_dir, f"{prefix}_info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info_dict, f, ensure_ascii=False, indent=2)

    faiss_path = os.path.join(output_dir, f"{prefix}_faiss_index.bin")
    faiss.write_index(faiss_index, faiss_path)

    vec_path = os.path.join(output_dir, f"{prefix}_tfidf_vectorizer.pkl")
    with open(vec_path, "wb") as f:
        pickle.dump(tfidf_vectorizer, f)

    mat_path = os.path.join(output_dir, f"{prefix}_tfidf_matrix.npz")
    sparse.save_npz(mat_path, tfidf_matrix)

    print(f"✅ Saved: {info_path}")
    print(f"✅ Saved: {faiss_path}")
    print(f"✅ Saved: {vec_path}")
    print(f"✅ Saved: {mat_path}")
    print(f"Embedding dim: {vectors.shape[1]} | #docs: {vectors.shape[0]}")


# =========================
# Knowledge CSV → 1 row = 1 chunk
# - JSON keeps ALL metadata columns
# - RAG text uses only topic_title + details
# =========================
REQUIRED_FOR_RAG = ["topic_title", "details"]


def _compose_rag_text(topic_title: str, details: str) -> str:
    topic = (topic_title or "").strip()
    det = (details or "").strip()

    if topic and det:
        return f"Topic: {topic}\\nDetails: {det}"
    if topic:
        return f"Topic: {topic}"
    if det:
        return f"Details: {det}"
    return ""


def load_from_knowledge_csv(csv_path: str) -> Dict[str, Dict[str, Any]]:
    """
    - Reads CSV (UTF-8-sig).
    - Each row becomes exactly 1 chunk (no splitting).
    - Metadata: ALL columns preserved.
    - Chunk text: ONLY topic_title + details.
    """
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    missing = [c for c in REQUIRED_FOR_RAG if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV missing required columns for RAG: {missing}. "
            f"Your CSV must include at least: {REQUIRED_FOR_RAG}"
        )

    info: Dict[str, Dict[str, Any]] = {}
    idx = 0

    for row_i in range(len(df)):
        # Preserve ALL columns as metadata
        rec: Dict[str, Any] = {}
        for col in df.columns:
            val = df.at[row_i, col]
            rec[col] = "" if pd.isna(val) else str(val)

        # Build RAG chunk from ONLY topic_title + details
        chunk = _compose_rag_text(
            topic_title=rec.get("topic_title", ""),
            details=rec.get("details", ""),
        )
        if not chunk.strip():
            continue

        rec["chunk"] = chunk
        rec["row_index"] = int(row_i)
        rec["chunk_idx"] = 0  # one chunk per row

        info[str(idx)] = rec
        idx += 1

    print(f"Prepared {len(info)} knowledge chunks from {len(df)} rows (1 row = 1 chunk).")
    return info


# =========================
# Orchestrator (knowledge_csv only here)
# =========================
def update_rag_database_from_knowledge_csv(
    knowledge_csv: str,
    output_dir: str,
    prefix: str,
    embed_model: str,
):
    info = load_from_knowledge_csv(knowledge_csv)

    if not info:
        print("⚠️ No documents prepared. Exiting.")
        return

    # Keep stable order: 0..N-1
    documents = [info[k]["chunk"] for k in sorted(info.keys(), key=lambda x: int(x))]

    vectors = _encode(documents, model_name=embed_model)
    index = _build_faiss(vectors)
    tfidf_vec, tfidf_mat = _build_tfidf(documents)

    _save_artifacts(
        output_dir=output_dir,
        prefix=prefix,
        info_dict=info,
        vectors=vectors,
        faiss_index=index,
        tfidf_vectorizer=tfidf_vec,
        tfidf_matrix=tfidf_mat,
    )

    print("First chunk:", documents[0][:200].replace("\\n", " "))
    print("🎉 RAG database update completed.")


# =========================
# Query / Retrieve (Hybrid)
# =========================
def load_rag_artifacts(output_dir: str, prefix: str):
    info_path = os.path.join(output_dir, f"{prefix}_info.json")
    faiss_path = os.path.join(output_dir, f"{prefix}_faiss_index.bin")
    vec_path = os.path.join(output_dir, f"{prefix}_tfidf_vectorizer.pkl")
    mat_path = os.path.join(output_dir, f"{prefix}_tfidf_matrix.npz")

    with open(info_path, "r", encoding="utf-8") as f:
        info: Dict[str, Dict[str, Any]] = json.load(f)

    index = faiss.read_index(faiss_path)

    with open(vec_path, "rb") as f:
        tfidf_vectorizer: TfidfVectorizer = pickle.load(f)

    tfidf_matrix = sparse.load_npz(mat_path)

    # stable order: 0..N-1
    keys = sorted(info.keys(), key=lambda x: int(x))
    return info, keys, index, tfidf_vectorizer, tfidf_matrix


def _dense_scores_faiss(
    query: str,
    model: SentenceTransformer,
    index: faiss.Index,
    top_k_dense: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      dense_idx: shape (m,) indices into your doc list
      dense_sim: shape (m,) cosine-like similarity in ~[-1,1]
    """
    q = model.encode([query], normalize_embeddings=True)
    q = np.asarray(q, dtype="float32")

    D, I = index.search(q, top_k_dense)  # shapes (1, k)
    D = D[0]
    I = I[0]

    # If vectors are unit-normalized, squared L2 = 2 - 2*cos => cos = 1 - D/2
    dense_sim = 1.0 - (D / 2.0)
    dense_sim = np.clip(dense_sim, -1.0, 1.0)
    return I, dense_sim


def _sparse_scores_tfidf(
    query: str,
    tfidf_vectorizer: TfidfVectorizer,
    tfidf_matrix: sparse.csr_matrix,
    top_k_sparse: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    TF-IDF uses L2 norm by default -> dot product ~ cosine similarity.
    Returns top indices and scores.
    """
    qv = tfidf_vectorizer.transform([query])          # (1, vocab)
    scores = (qv @ tfidf_matrix.T).toarray().ravel()  # (N,)

    if top_k_sparse >= len(scores):
        idx = np.argsort(-scores)
    else:
        idx = np.argpartition(-scores, top_k_sparse)[:top_k_sparse]
        idx = idx[np.argsort(-scores[idx])]

    return idx.astype(int), scores[idx].astype(float)


def hybrid_search(
    query: str,
    output_dir: str,
    prefix: str,
    embed_model_name: str,
    top_k: int = 5,
    top_k_dense: int = 50,
    top_k_sparse: int = 200,
    alpha: float = 0.6,
    method: str = "equal",  # "equal" | "weighted" | "rrf"
    rrf_k: int = 60,
) -> List[Dict[str, Any]]:
    """
    Hybrid retrieval from saved artifacts.

    method:
      - "equal": take top ceil(K/2) dense + top floor(K/2) sparse (dedupe)
      - "weighted": min-max normalize dense/sparse scores over union and combine with alpha
      - "rrf": reciprocal rank fusion on dense/sparse rankings

    Returns: list of dicts (metadata + chunk + score + retrieval)
    """
    info, keys, index, tfidf_vec, tfidf_mat = load_rag_artifacts(output_dir, prefix)
    model = SentenceTransformer(embed_model_name)

    # Get candidates (always fetch enough for fusion + equal split)
    k_dense_fetch = max(top_k_dense, top_k)
    k_sparse_fetch = max(top_k_sparse, top_k)

    dense_idx, dense_sim = _dense_scores_faiss(query, model, index, k_dense_fetch)
    sparse_idx, sparse_sim = _sparse_scores_tfidf(query, tfidf_vec, tfidf_mat, k_sparse_fetch)

    dense_idx_list = [int(i) for i in dense_idx.tolist()]
    dense_sim_list = [float(s) for s in dense_sim.tolist()]
    sparse_idx_list = [int(i) for i in sparse_idx.tolist()]
    sparse_sim_list = [float(s) for s in sparse_sim.tolist()]

    dense_map = {i: s for i, s in zip(dense_idx_list, dense_sim_list)}
    sparse_map = {i: s for i, s in zip(sparse_idx_list, sparse_sim_list)}

    # -------------------------
    # METHOD: EQUAL SPLIT
    # -------------------------
    if method == "equal":
        k_dense = math.ceil(top_k / 2)
        k_sparse = top_k // 2

        seen = set()
        results: List[Dict[str, Any]] = []

        # 1) dense first
        for i in dense_idx_list[:k_dense]:
            if i in seen:
                continue
            seen.add(i)
            rec = dict(info[keys[i]])
            rec["retrieval"] = "dense"
            rec["score"] = float(dense_map.get(i, 0.0))
            results.append(rec)
            if len(results) >= top_k:
                return results[:top_k]

        # 2) sparse
        for i in sparse_idx_list[:k_sparse]:
            if i in seen:
                continue
            seen.add(i)
            rec = dict(info[keys[i]])
            rec["retrieval"] = "sparse"
            rec["score"] = float(sparse_map.get(i, 0.0))
            results.append(rec)
            if len(results) >= top_k:
                return results[:top_k]

        # If dedupe reduced count, top up from remaining candidates
        for i in dense_idx_list[k_dense:]:
            if len(results) >= top_k:
                break
            if i in seen:
                continue
            seen.add(i)
            rec = dict(info[keys[i]])
            rec["retrieval"] = "dense"
            rec["score"] = float(dense_map.get(i, 0.0))
            results.append(rec)

        for i in sparse_idx_list[k_sparse:]:
            if len(results) >= top_k:
                break
            if i in seen:
                continue
            seen.add(i)
            rec = dict(info[keys[i]])
            rec["retrieval"] = "sparse"
            rec["score"] = float(sparse_map.get(i, 0.0))
            results.append(rec)

        return results[:top_k]

    # -------------------------
    # METHOD: RRF
    # -------------------------
    if method == "rrf":
        dense_rank = {i: r for r, i in enumerate(dense_idx_list, start=1)}
        sparse_rank = {i: r for r, i in enumerate(sparse_idx_list, start=1)}

        fused: List[Tuple[int, float]] = []
        for i in set(dense_rank.keys()) | set(sparse_rank.keys()):
            score = 0.0
            rd = dense_rank.get(i)
            rs = sparse_rank.get(i)
            if rd is not None:
                score += 1.0 / (rrf_k + rd)
            if rs is not None:
                score += 1.0 / (rrf_k + rs)
            fused.append((i, score))

        fused.sort(key=lambda x: x[1], reverse=True)

        results: List[Dict[str, Any]] = []
        for i, score in fused[:top_k]:
            rec = dict(info[keys[i]])
            rec["retrieval"] = "rrf"
            rec["score"] = float(score)
            results.append(rec)
        return results

    # -------------------------
    # METHOD: WEIGHTED
    # -------------------------
    if method == "weighted":
        cand = sorted(set(dense_map.keys()) | set(sparse_map.keys()))
        d_scores = np.array([dense_map.get(i, 0.0) for i in cand], dtype=float)
        s_scores = np.array([sparse_map.get(i, 0.0) for i in cand], dtype=float)

        def minmax(x: np.ndarray) -> np.ndarray:
            if x.size == 0:
                return x
            lo, hi = float(x.min()), float(x.max())
            if hi - lo < 1e-12:
                return np.zeros_like(x)
            return (x - lo) / (hi - lo)

        d_norm = minmax(d_scores)
        s_norm = minmax(s_scores)

        fused_scores = alpha * d_norm + (1.0 - alpha) * s_norm
        fused = sorted(zip(cand, fused_scores.tolist()), key=lambda x: x[1], reverse=True)

        results: List[Dict[str, Any]] = []
        for i, score in fused[:top_k]:
            rec = dict(info[keys[i]])
            rec["retrieval"] = "weighted"
            rec["score"] = float(score)
            results.append(rec)
        return results

    raise ValueError('method must be one of: "equal", "weighted", "rrf"')


# =========================
# Example
# =========================
if __name__ == "__main__":
    # 1) Build/update DB
    update_rag_database_from_knowledge_csv(
        knowledge_csv="./chunks.csv",
        output_dir="./RAG_database",
        prefix="knowledge",
        embed_model="all-MiniLM-L6-v2",
    )

    # 2) Query
    results = hybrid_search(
        query="reset password email not received",
        output_dir="./RAG_database",
        prefix="knowledge",
        embed_model_name="all-MiniLM-L6-v2",
        top_k=6,
        method="equal",
    )

    for r in results:
        print("\n---")
        print("retrieval:", r["retrieval"], "| score:", r["score"], "| row:", r.get("row_index"))
        print(r["chunk"][:250].replace("\n", " "))
