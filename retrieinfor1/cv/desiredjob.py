"""
Extract desired job position from CV data using embedding similarity.
Source field: 'Desired Job'

Vietnamese queries.
"""

import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from embedding_model import extract_field_by_similarity


def extract_desired_job(desired_job_text: str) -> str:
    """
    Extract desired job position from CV using embedding similarity.

    Args:
        desired_job_text: Raw 'Desired Job' field from user data (Vietnamese)

    Returns:
        Extracted desired job title/position in Vietnamese
    """
    if pd.isna(desired_job_text) or not str(desired_job_text).strip():
        return ""

    context = str(desired_job_text).strip()

    result = extract_field_by_similarity(context, "desired_job")

    if not result and len(context) < 100:
        return context.strip()

    return result if result else context.strip()


def process_cv_desired_job(input_csv: str, output_csv: str = None) -> pd.DataFrame:
    """
    Process CV data to extract desired job positions.
    """
    df = pd.read_csv(input_csv)

    desired_jobs = []
    for idx, row in df.iterrows():
        job = extract_desired_job(row.get('Desired Job', ''))
        desired_jobs.append({
            'UserID': row['UserID'],
            'desired_job': job
        })

    result_df = pd.DataFrame(desired_jobs)

    if output_csv:
        result_df.to_csv(output_csv, index=False)

    return result_df


if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent.parent
    input_path = base_dir / "huggingface + kaggle/dataset/kaggle/job-dataset-for-recommendation/USER_DATA_FINAL.csv"
    output_path = base_dir / "preprocessed/cv_desiredjob_temp.csv"

    result = process_cv_desired_job(str(input_path), str(output_path))
