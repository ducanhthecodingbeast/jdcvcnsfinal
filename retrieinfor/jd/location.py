"""
Extract job location from JD data using QA model.
Source field: 'Job Address'
"""

import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from qa_model import extract_multiple_answers


def extract_job_location(job_address: str) -> str:
    """
    Extract job location from JD.

    Args:
        job_address: Raw 'Job Address' field from job data

    Returns:
        Extracted job location
    """
    if pd.isna(job_address) or not str(job_address).strip():
        return ""

    context = str(job_address).strip()

    questions = [
        "Địa điểm làm việc của công việc này ở đâu?",
        "Thành phố hoặc địa chỉ của công việc là gì?",
        "Nơi làm việc cho vị trí này là ở đâu?"
    ]

    result = extract_multiple_answers(context, questions, combine=False)

    if not result and len(context) < 100:
        return context.strip()

    return result if result else context.strip()


def process_jd_location(input_csv: str, output_csv: str = None) -> pd.DataFrame:
    """
    Process JD data to extract job locations.

    Args:
        input_csv: Path to raw JOB_DATA_FINAL.csv
        output_csv: Optional path to save intermediate results

    Returns:
        DataFrame with JobID and location columns
    """
    df = pd.read_csv(input_csv)

    locations = []
    for idx, row in df.iterrows():
        loc = extract_job_location(row.get('Job Address', ''))
        locations.append({
            'JobID': row['JobID'],
            'job_location': loc
        })

    result_df = pd.DataFrame(locations)

    if output_csv:
        result_df.to_csv(output_csv, index=False)

    return result_df


if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent.parent
    input_path = base_dir / "huggingface + kaggle/dataset/kaggle/job-dataset-for-recommendation/JOB_DATA_FINAL.csv"
    output_path = base_dir / "preprocessed/jd_location_temp.csv"

    result = process_jd_location(str(input_path), str(output_path))