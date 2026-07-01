"""
Extract summary/objective from CV data using multilingual embeddings.
"""

import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from embedding_model import extract_field_by_similarity


def extract_summary(target_text: str) -> str:
    if pd.isna(target_text) or not str(target_text).strip():
        return ""
    return extract_field_by_similarity(
        str(target_text),
        "mục tiêu nghề nghiệp hoặc tóm tắt kinh nghiệm làm việc"
    )


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