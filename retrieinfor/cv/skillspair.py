"""
Extract skills from CV data using QA model.
Source field: 'Skills' (free-text skills description)

All questions are in Vietnamese for Vietnamese project data.
"""

import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from qa_model import extract_multiple_answers


def extract_skills(skills_text: str) -> str:
    """
    Extract skills from CV skills text.

    Args:
        skills_text: Raw 'Skills' field from user data (Vietnamese)

    Returns:
        Extracted skills as semicolon-separated string in Vietnamese
    """
    if pd.isna(skills_text) or not str(skills_text).strip():
        return ""

    context = str(skills_text).strip()

    # Multiple targeted questions in Vietnamese
    questions = [
        "Người này có kỹ năng kỹ thuật hoặc phần mềm nào?",
        "Các kỹ năng chuyên môn hoặc năng lực được đề cập là gì?",
        "Người này có chứng chỉ hoặc bằng cấp gì?",
        "Khả năng chính và chuyên môn của người này là gì?",
        "Người này có thể sử dụng công cụ, công nghệ hoặc phương pháp nào?",
        "Các ngôn ngữ hoặc kỹ năng chuyên biệt nào được liệt kê?"
    ]

    return extract_multiple_answers(context, questions, combine=True)


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