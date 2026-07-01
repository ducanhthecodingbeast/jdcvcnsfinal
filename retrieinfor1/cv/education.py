"""
Extract education information from CV using embedding similarity.

All queries are in Vietnamese for Vietnamese project data.
"""

import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from embedding_model import extract_field_by_similarity


def extract_education(degree_text: str) -> dict:
    """
    Extract education information from CV using embedding similarity.

    Args:
        degree_text: Raw 'Degree' field from user data (Vietnamese)

    Returns:
        Dict with education_level, institution, major in Vietnamese
    """
    if pd.isna(degree_text) or not str(degree_text).strip():
        return {'education_level': '', 'institution': '', 'major': ''}

    context = str(degree_text)

    education_level = extract_field_by_similarity(context, "education_level")
    institution = extract_field_by_similarity(context, "institution")
    major = extract_field_by_similarity(context, "major")

    return {
        'education_level': education_level,
        'institution': institution,
        'major': major
    }


def process_cv_education(input_csv: str, output_csv: str = None) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    records = []
    for idx, row in df.iterrows():
        edu = extract_education(row.get('Degree', ''))
        edu['UserID'] = row['UserID']
        records.append(edu)
    result_df = pd.DataFrame(records)
    if output_csv:
        result_df.to_csv(output_csv, index=False)
    return result_df


if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent.parent
    input_path = base_dir / "huggingface + kaggle/dataset/kaggle/job-dataset-for-recommendation/USER_DATA_FINAL.csv"
    output_path = base_dir / "preprocessed/cv_education_temp.csv"
    result = process_cv_education(str(input_path), str(output_path))
