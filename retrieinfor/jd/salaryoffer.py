"""
Extract salary offer information from JD data using QA model.
Source field: 'Salary'
"""

import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from qa_model import extract_multiple_answers


def extract_salary_offer(salary_text: str) -> str:
    """
    Extract salary offer from JD.

    Args:
        salary_text: Raw 'Salary' field from job data

    Returns:
        Extracted salary offer information
    """
    if pd.isna(salary_text) or not str(salary_text).strip():
        return ""

    context = str(salary_text).strip()

    # Direct extraction - field is already salary info
    questions = [
        "Mức lương được đề xuất cho công việc này là bao nhiêu?",
        "Khoảng lương hoặc đãi ngộ là gì?",
        "Mức thù lao được đưa ra là bao nhiêu?",
        "Lương hàng tháng hoặc hàng năm là bao nhiêu?"
    ]

    # Try QA extraction
    result = extract_multiple_answers(context, questions, combine=False)

    # If QA doesn't extract, return cleaned raw text if reasonable length
    if not result and len(context) < 100:
        return context.strip()

    return result if result else context.strip()


def process_jd_salary(input_csv: str, output_csv: str = None) -> pd.DataFrame:
    """
    Process JD data to extract salary offers.

    Args:
        input_csv: Path to raw JOB_DATA_FINAL.csv
        output_csv: Optional path to save intermediate results

    Returns:
        DataFrame with JobID and salary columns
    """
    df = pd.read_csv(input_csv)

    salaries = []
    for idx, row in df.iterrows():
        sal = extract_salary_offer(row.get('Salary', ''))
        salaries.append({
            'JobID': row['JobID'],
            'salary_offer': sal
        })

    result_df = pd.DataFrame(salaries)

    if output_csv:
        result_df.to_csv(output_csv, index=False)

    return result_df


if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent.parent
    input_path = base_dir / "huggingface + kaggle/dataset/kaggle/job-dataset-for-recommendation/JOB_DATA_FINAL.csv"
    output_path = base_dir / "preprocessed/jd_salary_temp.csv"

    result = process_jd_salary(str(input_path), str(output_path)) 