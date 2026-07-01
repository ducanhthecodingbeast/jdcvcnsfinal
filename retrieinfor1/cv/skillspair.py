"""
Extract skills from CV data using embedding similarity.
Source field: 'Skills' (free-text skills description)

Uses query variation for diversity. Vietnamese queries.
"""

import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from embedding_model import extract_field_by_similarity


def extract_skills(skills_text: str) -> str:
    """
    Extract skills from CV skills text using embedding similarity.

    Args:
        skills_text: Raw 'Skills' field from user data (Vietnamese)

    Returns:
        Extracted skills as semicolon-separated string in Vietnamese
    """
    if pd.isna(skills_text) or not str(skills_text).strip():
        return ""

    context = str(skills_text).strip()

    # Use field key "skills" which maps to multiple query variants
    return extract_field_by_similarity(context, "skills")


def process_cv_skills(input_csv: str, output_csv: str = None) -> pd.DataFrame:
    """
    Process CV data to extract skills.
    """
    df = pd.read_csv(input_csv)

    skills_records = []
    for idx, row in df.iterrows():
        skills = extract_skills(row.get('Skills', ''))
        skills_records.append({
            'UserID': row['UserID'],
            'skills': skills
        })

    result_df = pd.DataFrame(skills_records)

    if output_csv:
        result_df.to_csv(output_csv, index=False)

    return result_df


if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent.parent
    input_path = base_dir / "huggingface + kaggle/dataset/kaggle/job-dataset-for-recommendation/USER_DATA_FINAL.csv"
    output_path = base_dir / "preprocessed/cv_skills_temp.csv"

    result = process_cv_skills(str(input_path), str(output_path))