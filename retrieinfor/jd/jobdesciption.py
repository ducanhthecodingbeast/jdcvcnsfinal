"""
Extract job description, responsibilities, and requirements from JD data using QA model.
Source fields: 'Job Description', 'Job Requirements'

All questions are in Vietnamese for Vietnamese project data.
"""

import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from qa_model import extract_multiple_answers


def extract_job_description(job_desc: str, job_req: str) -> dict:
    """
    Extract job description and key responsibilities from JD.

    Args:
        job_desc: Raw 'Job Description' field (Vietnamese)
        job_req: Raw 'Job Requirements' field (Vietnamese)

    Returns:
        Dict with job_description, responsibilities, requirements in Vietnamese
    """
    # Combine both fields for richer context
    context = ""
    if pd.notna(job_desc):
        context += str(job_desc).strip() + " "
    if pd.notna(job_req):
        context += str(job_req).strip()

    if not context.strip():
        return {
            'job_description': '',
            'responsibilities': '',
            'requirements': ''
        }

    # Extract main job description/title
    job_description = extract_multiple_answers(context, [
        "Công việc chính hoặc vị trí được tuyển dụng là gì?",
        "Công việc này là về gì?",
        "Công ty đang tuyển dụng vai trò nào?",
        "Tên công việc hoặc tên vị trí là gì?"
    ], combine=False)

    # Extract responsibilities
    responsibilities = extract_multiple_answers(context, [
        "Các nhiệm vụ và trách nhiệm chính của công việc là gì?",
        "Nhân viên sẽ thực hiện những nhiệm vụ nào?",
        "Người này sẽ làm gì hàng ngày?",
        "Trách nhiệm chính của vai trò này là gì?"
    ], combine=True)

    # Extract requirements (education, experience, skills needed)
    requirements = extract_multiple_answers(context, [
        "Yêu cầu cho công việc này là gì?",
        "Ứng viên cần có những bằng cấp gì?",
        "Kinh nghiệm hoặc kỹ năng nào được yêu cầu?",
        "Học vấn hoặc nền tảng nào là cần thiết?"
    ], combine=True)

    return {
        'job_description': job_description,
        'responsibilities': responsibilities,
        'requirements': requirements
    }


def process_jd_description(input_csv: str, output_csv: str = None) -> pd.DataFrame:
    """
    Process JD data to extract job descriptions.
    """
    df = pd.read_csv(input_csv)

    descriptions = []
    for idx, row in df.iterrows():
        desc = extract_job_description(
            row.get('Job Description', ''),
            row.get('Job Requirements', '')
        )
        desc['JobID'] = row['JobID']
        descriptions.append(desc)

    result_df = pd.DataFrame(descriptions)

    if output_csv:
        result_df.to_csv(output_csv, index=False)

    return result_df


if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent.parent
    input_path = base_dir / "huggingface + kaggle/dataset/kaggle/job-dataset-for-recommendation/JOB_DATA_FINAL.csv"
    output_path = base_dir / "preprocessed/jd_description_temp.csv"

    result = process_jd_description(str(input_path), str(output_path))