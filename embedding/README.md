# Embedding Modules

Two separate embedding pipelines for CV-JD similarity scoring.

## bgme3.py

**Model**: BAAI/bge-m3 (multilingual dense embeddings)

**Comparison**:
- JD side: `required_skills` (extracted from Job Requirements + Job Description)
- CV side: `skills` + `summary` (combined)

**Output**: `bge_similarity_scores.csv` with columns `JobID`, `UserID`, `bge_similarity_score`

## jobbert.py

**Model**: TechWolf/JobBERT-v2 (job domain embeddings)

**Comparison**:
- JD side: `job_description` (extracted job title/role)
- CV side: `skills` (extracted skills pairs)

**Output**: `jobbert_similarity_scores.csv` with columns `JobID`, `UserID`, `jobbert_similarity_score`

## Usage

```bash
# Run BGE-M3 scoring
python embedding/bgme3.py

# Run JobBERT-v2 scoring
python embedding/jobbert.py
```

Both read from `preprocessed/cv.csv` and `preprocessed/jd.csv` (must be generated first via `retrieinfor/main.py`).

Models are cached to `dataset/model/` on first run. GPU is used automatically if available.