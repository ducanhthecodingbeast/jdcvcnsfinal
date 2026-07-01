"""
Shared embedding model loader for semantic information retrieval from CV and JD texts.
Uses sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (multilingual, good for Vietnamese).

Strategy:
- Split raw text into sentences/chunks.
- Embed the chunks + embed label queries in Vietnamese.
- Return the most semantically similar chunks for each field label.
"""

import torch
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance
_embedding_model = None

def get_embedding_model():
    """Load and return the multilingual embedding model (singleton)."""
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading embedding model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _embedding_model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            device=device
        )
        logger.info(f"Embedding model loaded on device: {device}")
    return _embedding_model

def split_into_chunks(text: str, max_chunk_size: int = 200) -> List[str]:
    """
    Split text into sentences or small chunks for semantic search.
    """
    if not text or not isinstance(text, str):
        return []

    text = text.strip()
    # Split by common Vietnamese/English sentence separators
    sentences = re.split(r'[.!?;\n•\-]+', text)
    chunks = []
    current = ""

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if len(current) + len(sent) + 1 <= max_chunk_size:
            current = (current + " " + sent).strip()
        else:
            if current:
                chunks.append(current)
            current = sent

    if current:
        chunks.append(current)

    return [c for c in chunks if len(c) > 5]  # filter very short noise

def extract_field_by_similarity(
    text: str,
    query: str,
    top_k: int = 2,
    min_score: float = 0.25
) -> str:
    """
    Given raw text and a Vietnamese query (e.g. "mục tiêu nghề nghiệp"),
    return the top semantically similar chunks joined by " ; ".
    """
    model = get_embedding_model()
    chunks = split_into_chunks(text)

    if not chunks:
        return ""

    # Embed
    chunk_embs = model.encode(chunks, convert_to_tensor=True, normalize_embeddings=True)
    query_emb = model.encode([query], convert_to_tensor=True, normalize_embeddings=True)

    # Cosine similarity
    scores = torch.nn.functional.cosine_similarity(query_emb, chunk_embs)

    # Get top results above threshold
    top_scores, top_indices = torch.topk(scores, k=min(top_k, len(chunks)))

    selected = []
    for score, idx in zip(top_scores, top_indices):
        if score.item() >= min_score:
            selected.append(chunks[idx])

    if not selected:
        # Fallback: return the highest scoring chunk even if below threshold
        best_idx = torch.argmax(scores).item()
        return chunks[best_idx]

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for s in selected:
        s_lower = s.lower()
        if s_lower not in seen:
            seen.add(s_lower)
            unique.append(s)

    return " ; ".join(unique)

def extract_multiple_fields(
    text: str,
    queries: Dict[str, str]
) -> Dict[str, str]:
    """
    Extract multiple fields at once.
    queries = {"summary": "mục tiêu nghề nghiệp", "skills": "kỹ năng chuyên môn", ...}
    """
    results = {}
    for field, query in queries.items():
        results[field] = extract_field_by_similarity(text, query)
    return results

# Predefined Vietnamese queries for our schema
CV_QUERIES = {
    "summary": "mục tiêu nghề nghiệp hoặc tóm tắt kinh nghiệm",
    "education_level": "trình độ học vấn bằng cấp đại học cao đẳng",
    "institution": "trường đại học cao đẳng nơi đào tạo",
    "major": "chuyên ngành học thuật",
    "skills": "kỹ năng chuyên môn kinh nghiệm làm việc"
}

JD_QUERIES = {
    "job_description": "mô tả công việc vị trí tuyển dụng",
    "responsibilities": "nhiệm vụ trách nhiệm công việc",
    "requirements": "yêu cầu kinh nghiệm kỹ năng",
    "required_education_level": "yêu cầu trình độ học vấn",
    "required_major": "yêu cầu chuyên ngành",
    "required_skills": "kỹ năng yêu cầu cho công việc"
}