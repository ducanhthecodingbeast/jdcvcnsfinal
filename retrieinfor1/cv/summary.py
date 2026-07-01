"""
Extract summary/objective from CV data using embedding similarity.
Source field: 'Target'

Uses query variation for diversity. Vietnamese queries.
"""

import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from embedding_model import extract_field_by_similarity


def extract_summary(target_text: str) -> str:
    """
    Extract career objective or experience summary from CV using embedding similarity.

    Args:
        target_text: Raw 'Target' field from user data (Vietnamese)

    Returns:
        Extracted summary/objective in Vietnamese (diverse chunks joined by " ; ")
    """
    if pd.isna(target_text) or not str(target_text).strip():
        return ""

    context = str(target_text).strip()

    # Use field key "summary" which maps to query variants internally
    return extract_field_by_similarity(context, "summary")


def process_cv_summary(input_csv: str, output_csv: str = None) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    summaries = []
    for idx, row in df.iterrows():
        summary = extract_summary(row.get('Target', ''))
        summaries.append({'UserID': row['UserID'], 'summary': summary})
    result_df = pd.DataFrame(summaries)
    if output_csv:
        result_df.to_csv(output_csv, index=False)
    return result_df


if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent.parent
    input_path = base_dir / "huggingface + kaggle/dataset/kaggle/job-dataset-for-recommendation/USER_DATA_FINAL.csv"
    output_path = base_dir / "preprocessed/cv_summary_temp.csv"
    result = process_cv_summary(str(input_path), str(output_path))
