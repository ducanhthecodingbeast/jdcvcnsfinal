"""
Extract salary expectation from CV data using QA model.
Source field: 'Desired Salary'

All questions are in Vietnamese for Vietnamese project data.
"""

import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from qa_model import extract_multiple_answers


def extract_salary(salary_text: str) -> str:
    """
    Extract salary expectation from CV.

    Args:
        salary_text: Raw 'Desired Salary' field from user data (Vietnamese)

    Returns:
        Extracted salary information in Vietnamese
    """
    if pd.isna(salary_text) or not str(salary_text).strip():
        return ""

    context = str(salary_text).strip()

    questions = [
        "Mức lương mong muốn của người này là bao nhiêu?",
        "Người này kỳ vọng mức lương như thế nào?",
        "Mức thù lao mong muốn là bao nhiêu?",
        "Lương kỳ vọng hàng tháng hoặc hàng năm là bao nhiêu?"
    ]

    result = extract_multiple_answers(context, questions, combine=False)

    if not result and len(context) < 50:
        return context.strip()

    return result if result else context.strip()


def process_cv_salary(input_csv: str, output_csv: str = None) -> pd.DataFrame:
    """
    Process CV data to extract salary expectations.
    """
    df = pd.read_csv(input_csv)

    salaries = []
    for idx, row in df.iterrows():
        sal = extract_salary(row.get('Desired Salary', ''))
        salaries.append({
            'UserID': row['UserID'],
            'salary_expectation': sal
        })

    result_df = pd.DataFrame(salaries)

    if output_csv:
        result_df.to_csv(output_csv, index=False)

    return result_df


if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent.parent
    input_path = base_dir / "huggingface + kaggle/dataset/kaggle/job-dataset-for-recommendation/USER_DATA_FINAL.csv"
    output_path = base_dir / "preprocessed/cv_salary_temp.csv"

    result = process_cv_salary(str(input_path), str(output_path))
