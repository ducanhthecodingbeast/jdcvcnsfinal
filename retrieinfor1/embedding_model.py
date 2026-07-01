"""
Shared embedding model loader for semantic information retrieval from CV and JD texts.
Uses BAAI/bge-m3 (highest quality multilingual retrieval) with paraphrase-multilingual fallback.

Strategy:
- Split raw text into sentences/chunks.
- Use query variation (multiple Vietnamese phrasings per field) for diversity.
- Embed chunks + queries, select top matches per variant, union + dedup.
- Return semantically relevant but non-redundant chunks joined by " ; ".
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

# Primary model: BAAI/bge-m3 (best multilingual retrieval per downloads + quality)
# Fallback: paraphrase-multilingual-MiniLM-L12-v2 (lighter, still good multilingual)
PRIMARY_MODEL = "BAAI/bge-m3"
FALLBACK_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def get_embedding_model():
    """Load and return the multilingual embedding model (singleton)."""
    global _embedding_model
    if _embedding_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = PRIMARY_MODEL
        try:
            logger.info(f"Loading embedding model: {model_name}")
            _embedding_model = SentenceTransformer(model_name, device=device)
            logger.info(f"Embedding model loaded on device: {device}")
        except Exception as e:
            logger.warning(f"Failed to load {model_name}: {e}. Falling back to {FALLBACK_MODEL}")
            _embedding_model = SentenceTransformer(FALLBACK_MODEL, device=device)
            logger.info(f"Fallback model loaded on device: {device}")
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

# Query variants per field for diversity (Vietnamese-focused, 2-3 per field)
# Strategy: query variation + union + dedup ensures we capture different semantic angles
FIELD_QUERIES: Dict[str, List[str]] = {
    # CV fields
    "summary": [
        "mục tiêu nghề nghiệp",
        "tóm tắt kinh nghiệm làm việc",
        "điểm mạnh giá trị cá nhân"
    ],
    "education_level": [
        "trình độ học vấn",
        "bằng cấp"
    ],
    "institution": [
        "trường đại học",
        "nơi đào tạo"
    ],
    "major": [
        "chuyên ngành",
        "ngành học"
    ],
    "skills": [
        "kỹ năng chuyên môn",
        "kinh nghiệm làm việc",
        "năng lực kỹ thuật"
    ],
    # JD fields
    "job_description": [
        "mô tả công việc",
        "vị trí tuyển dụng"
    ],
    "responsibilities": [
        "nhiệm vụ",
        "trách nhiệm công việc"
    ],
    "requirements": [
        "yêu cầu công việc",
        "kinh nghiệm cần có",
        "kỹ năng yêu cầu"
    ],
    "required_education_level": [
        "yêu cầu trình độ",
        "bằng cấp cần thiết"
    ],
    "required_major": [
        "yêu cầu chuyên ngành",
        "ngành học ưu tiên"
    ],
    "required_skills": [
        "kỹ năng yêu cầu",
        "công nghệ cần biết",
        "năng lực bắt buộc"
    ],
    # Additional fields for salary and location (both CV and JD)
    "salary_expectation": [
        "mức lương mong muốn",
        "lương kỳ vọng"
    ],
    "salary_offer": [
        "mức lương đề xuất",
        "khoảng lương"
    ],
    "location": [
        "địa điểm làm việc",
        "nơi làm việc mong muốn"
    ],
    "job_location": [
        "địa điểm công việc",
        "địa chỉ làm việc"
    ],
    "desired_job": [
        "vị trí công việc mong muốn",
        "loại công việc tìm kiếm"
    ]
}

def extract_field_by_similarity(
    text: str,
    query: str,
    top_k: int = 2,
    min_score: float = 0.25
) -> str:
    """
    Given raw text and a field key (e.g. "summary"), use query variation to extract
    diverse semantically relevant chunks. Returns chunks joined by " ; ".

    query: either a known field key (looks up variants) or a raw Vietnamese query string.
    """
    model = get_embedding_model()
    chunks = split_into_chunks(text)

    if not chunks:
        return ""

    # Short-circuit: very few chunks, no diversity needed
    if len(chunks) <= 3:
        # Use first variant if known key, else raw query
        variants = FIELD_QUERIES.get(query, [query])
        q0 = variants[0]
        chunk_embs = model.encode(chunks, convert_to_tensor=True, normalize_embeddings=True)
        query_emb = model.encode([q0], convert_to_tensor=True, normalize_embeddings=True)
        scores = torch.nn.functional.cosine_similarity(query_emb, chunk_embs)
        top_scores, top_indices = torch.topk(scores, k=min(1, len(chunks)))
        for score, idx in zip(top_scores, top_indices):
            if score.item() >= min_score:
                return chunks[idx]
        # fallback to best
        return chunks[torch.argmax(scores).item()]

    # Determine query variants
    variants = FIELD_QUERIES.get(query, [query])

    # Embed all chunks once
    chunk_embs = model.encode(chunks, convert_to_tensor=True, normalize_embeddings=True)

    selected = []
    for q in variants:
        q_emb = model.encode([q], convert_to_tensor=True, normalize_embeddings=True)
        scores = torch.nn.functional.cosine_similarity(q_emb, chunk_embs)
        top_scores, top_indices = torch.topk(scores, k=min(top_k, len(chunks)))
        for score, idx in zip(top_scores, top_indices):
            if score.item() >= min_score:
                selected.append(chunks[idx])

    if not selected:
        # Fallback: best chunk from first variant
        q0 = variants[0]
        q_emb = model.encode([q0], convert_to_tensor=True, normalize_embeddings=True)
        scores = torch.nn.functional.cosine_similarity(q_emb, chunk_embs)
        best_idx = torch.argmax(scores).item()
        return chunks[best_idx]

    # Deduplicate while preserving order (case-insensitive)
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

# Legacy single-string query dicts kept for backward compatibility with any external callers.
# New code should use FIELD_QUERIES (above) via extract_field_by_similarity(field_key).
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