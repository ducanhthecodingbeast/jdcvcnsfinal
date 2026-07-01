"""
BGE-M3 dense vector embeddings for job-CV similarity scoring.

Compares JD required skills against CV skills + summary using BAAI/bge-m3.
Input data from retrieinfor extraction pipeline (preprocessed/cv.csv, preprocessed/jd.csv).
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
import logging
from typing import List, Tuple, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration
BGE_MODEL_NAME = "BAAI/bge-m3"
MODEL_CACHE_DIR = Path(__file__).parent.parent / "dataset" / "model" / "bge-m3"


def load_bge_model() -> SentenceTransformer:
    """
    Load BAAI/bge-m3 model for dense embeddings.
    Uses local cache if available, otherwise downloads.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading BGE-M3 model on device: {device}")

    if MODEL_CACHE_DIR.exists():
        logger.info(f"Loading from local cache: {MODEL_CACHE_DIR}")
        model = SentenceTransformer(str(MODEL_CACHE_DIR), device=device)
    else:
        logger.info(f"Downloading model: {BGE_MODEL_NAME}")
        model = SentenceTransformer(BGE_MODEL_NAME, device=device)

    logger.info("BGE-M3 model loaded successfully")
    return model


def prepare_jd_text(required_skills: str) -> str:
    """
    Prepare JD skills text for embedding.
    """
    if pd.isna(required_skills) or not str(required_skills).strip():
        return ""
    return str(required_skills).strip()


def prepare_cv_text(skills: str, summary: str) -> str:
    """
    Prepare CV text by combining skills and summary for embedding.
    """
    parts = []
    if pd.notna(skills) and str(skills).strip():
        parts.append(f"Skills: {str(skills).strip()}")
    if pd.notna(summary) and str(summary).strip():
        parts.append(f"Summary: {str(summary).strip()}")
    return " ".join(parts)


def compute_embeddings(
    texts: List[str],
    model: SentenceTransformer,
    batch_size: int = 32
) -> np.ndarray:
    """
    Compute dense embeddings for a list of texts.

    Args:
        texts: List of text strings to embed
        model: Loaded SentenceTransformer model
        batch_size: Batch size for encoding

    Returns:
        numpy array of shape (n_texts, embedding_dim)
    """
    # Filter empty texts, keep track of indices
    valid_indices = [i for i, t in enumerate(texts) if t.strip()]
    valid_texts = [texts[i] for i in valid_indices]

    if not valid_texts:
        return np.array([])

    logger.info(f"Computing embeddings for {len(valid_texts)} texts")
    embeddings = model.encode(
        valid_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # L2 normalize for cosine similarity
    )

    # Create full embedding matrix with zeros for empty texts
    full_embeddings = np.zeros((len(texts), embeddings.shape[1]))
    for idx, emb_idx in enumerate(valid_indices):
        full_embeddings[emb_idx] = embeddings[idx]

    return full_embeddings


def compute_similarity_scores(
    jd_embeddings: np.ndarray,
    cv_embeddings: np.ndarray
) -> np.ndarray:
    """
    Compute cosine similarity between JD and CV embeddings.

    Args:
        jd_embeddings: Shape (n_jds, dim)
        cv_embeddings: Shape (n_cvs, dim)

    Returns:
        Similarity matrix of shape (n_jds, n_cvs)
    """
    if jd_embeddings.size == 0 or cv_embeddings.size == 0:
        return np.array([])

    logger.info(f"Computing similarity: {len(jd_embeddings)} JDs x {len(cv_embeddings)} CVs")
    similarity_matrix = cosine_similarity(jd_embeddings, cv_embeddings)
    return similarity_matrix


def process_similarity(
    jd_df: pd.DataFrame,
    cv_df: pd.DataFrame,
    model: Optional[SentenceTransformer] = None
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Process JD and CV dataframes to compute similarity scores.

    Args:
        jd_df: DataFrame with JobID and required_skills columns
        cv_df: DataFrame with UserID, skills, and summary columns
        model: Optional pre-loaded model (loads if not provided)

    Returns:
        Tuple of (scores_df, similarity_matrix)
        scores_df has columns: JobID, UserID, bge_similarity_score
    """
    if model is None:
        model = load_bge_model()

    # Prepare texts
    logger.info("Preparing texts for embedding...")
    jd_texts = jd_df['required_skills'].apply(prepare_jd_text).tolist()
    cv_texts = [
        prepare_cv_text(row.get('skills', ''), row.get('summary', ''))
        for _, row in cv_df.iterrows()
    ]

    # Compute embeddings
    jd_embeddings = compute_embeddings(jd_texts, model)
    cv_embeddings = compute_embeddings(cv_texts, model)

    # Compute similarities
    similarity_matrix = compute_similarity_scores(jd_embeddings, cv_embeddings)

    # Create scores dataframe (long format)
    scores = []
    for jd_idx, job_id in enumerate(jd_df['JobID']):
        for cv_idx, user_id in enumerate(cv_df['UserID']):
            if jd_idx < len(similarity_matrix) and cv_idx < len(similarity_matrix[jd_idx]):
                scores.append({
                    'JobID': job_id,
                    'UserID': user_id,
                    'bge_similarity_score': float(similarity_matrix[jd_idx, cv_idx])
                })

    scores_df = pd.DataFrame(scores)
    logger.info(f"Computed {len(scores_df)} similarity scores")

    return scores_df, similarity_matrix


def main():
    """
    Main entry point for BGE-M3 similarity computation.
    Reads from preprocessed data and outputs similarity scores.
    """
    project_root = Path(__file__).parent.parent
    cv_path = project_root / "preprocessed" / "cv.csv"
    jd_path = project_root / "preprocessed" / "jd.csv"
    output_path = project_root / "preprocessed" / "bge_similarity_scores.csv"

    logger.info("=" * 60)
    logger.info("BGE-M3 Similarity Computation")
    logger.info("=" * 60)

    # Load data
    logger.info(f"Loading CV data from {cv_path}")
    cv_df = pd.read_csv(cv_path)
    logger.info(f"Loaded {len(cv_df)} CV records")

    logger.info(f"Loading JD data from {jd_path}")
    jd_df = pd.read_csv(jd_path)
    logger.info(f"Loaded {len(jd_df)} JD records")

    # Compute similarities
    scores_df, _ = process_similarity(jd_df, cv_df)

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scores_df.to_csv(output_path, index=False)
    logger.info(f"Saved similarity scores to {output_path}")

    # Summary stats
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total scores computed: {len(scores_df)}")
    if len(scores_df) > 0:
        logger.info(f"Score range: [{scores_df['bge_similarity_score'].min():.4f}, {scores_df['bge_similarity_score'].max():.4f}]")
        logger.info(f"Mean score: {scores_df['bge_similarity_score'].mean():.4f}")


if __name__ == "__main__":
    main()
