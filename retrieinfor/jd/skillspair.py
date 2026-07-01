"""
Extract required skills from JD data using QA model.
Source fields: 'Job Requirements', 'Job Description'
"""

import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from qa_model import extract_multiple_answers


def extract_required_skills(job_req: str, job_desc: str = "") -> str:
    """
    Extract required skills from JD.

    Args:
        job_req: Raw 'Job Requirements' field
        job_desc: Optional 'Job Description' field

    Returns:
        Extracted required skills as semicolon-separated string
    """
    # Combine both fields
    context = ""
    if pd.notna(job_req):
        context += str(job_req).strip() + " "
    if pd.notna(job_desc):
        context += str(job_desc).strip()

    if not context.strip():
        return ""

    # Extract various types of required skills
    questions = [
        "Yêu cầu kỹ năng kỹ thuật hoặc phần mềm nào cho công việc này?",
        "Yêu cầu kỹ năng chuyên môn hoặc năng lực gì?",
        "Yêu cầu kinh nghiệm hoặc bằng cấp gì?",
        "Ứng viên phải biết công cụ, công nghệ hoặc phương pháp nào?",
        "Yêu cầu chứng chỉ hoặc giấy phép gì?",
        "Yêu cầu ngôn ngữ hoặc kỹ năng chuyên biệt nào?"
    ]

    return extract_multiple_answers(context, questions, combine=True)


def process_jd_skills(input_csv: str, output_csv: str = None) -> pd.DataFrame:
    """
    Process JD data to extract required skills.

    Args:
        input_csv: Path to raw JOB_DATA_FINAL.csv
        output_csv: Optional path to save intermediate results

    Returns:
        DataFrame with JobID and skills columns
    """
    df = pd.read_csv(input_csv)

    skills_records = []
    for idx, row in df.iterrows():
        skills = extract_required_skills(
            row.get('Job Requirements', ''),
            row.get('Job Description', '')
        )
        skills_records.append({
            'JobID': row['JobID'],
            'required_skills': skills
        })

    result_df = pd.DataFrame(skills_records)

    if output_csv:
        result_df.to_csv(output_csv, index=False)

    return result_df


if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent.parent
    input_path = base_dir / "huggingface + kaggle/dataset/kaggle/job-dataset-for-recommendation/JOB_DATA_FINAL.csv"
    output_path = base_dir / "preprocessed/jd_skills_temp.csv"

    result = process_jd_skills(str(input_path), str(output_path))