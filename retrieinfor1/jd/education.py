"""
Extract education requirements from JD data using embedding similarity.
Source field: 'Job Requirements'
"""

import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from embedding_model import extract_field_by_similarity


def extract_education_requirements(job_req: str, job_desc: str = "") -> dict:
    """
    Extract education requirements from JD using embedding similarity.

    Args:
        job_req: Raw 'Job Requirements' field
        job_desc: Optional 'Job Description' field for additional context

    Returns:
        Dict with required_education_level, required_major
    """
    # Combine both fields
    context = ""
    if pd.notna(job_req):
        context += str(job_req).strip() + " "
    if pd.notna(job_desc):
        context += str(job_desc).strip()

    if not context.strip():
        return {
            'required_education_level': '',
            'required_major': ''
        }

    required_education_level = extract_field_by_similarity(context, "required_education_level")
    required_major = extract_field_by_similarity(context, "required_major")

    return {
        'required_education_level': required_education_level,
        'required_major': required_major
    }


def process_jd_education(input_csv: str, output_csv: str = None) -> pd.DataFrame:
    """
    Process JD data to extract education requirements.

    Args:
        input_csv: Path to raw JOB_DATA_FINAL.csv
        output_csv: Optional path to save intermediate results

    Returns:
        DataFrame with JobID and education requirement columns
    """
    df = pd.read_csv(input_csv)

    education_records = []
    for idx, row in df.iterrows():
        edu = extract_education_requirements(
            row.get('Job Requirements', ''),
            row.get('Job Description', '')
        )
        edu['JobID'] = row['JobID']
        education_records.append(edu)

    result_df = pd.DataFrame(education_records)

    if output_csv:
        result_df.to_csv(output_csv, index=False)

    return result_df


if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent.parent
    input_path = base_dir / "huggingface + kaggle/dataset/kaggle/job-dataset-for-recommendation/JOB_DATA_FINAL.csv"
    output_path = base_dir / "preprocessed/jd_education_temp.csv"

    result = process_jd_education(str(input_path), str(output_path))