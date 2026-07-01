"""
Extract location information from CV data using QA model.
Source field: 'Workplace Desired'

All questions are in Vietnamese for Vietnamese project data.
"""

import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from qa_model import extract_multiple_answers


def extract_location(workplace_text: str) -> str:
    """
    Extract desired workplace location from CV.

    Args:
        workplace_text: Raw 'Workplace Desired' field from user data (Vietnamese)

    Returns:
        Extracted location in Vietnamese
    """
    if pd.isna(workplace_text) or not str(workplace_text).strip():
        return ""

    context = str(workplace_text).strip()

    questions = [
        "Người này muốn làm việc tại thành phố hoặc địa điểm nào?",
        "Địa điểm làm việc mong muốn là gì?",
        "Ứng viên muốn làm việc ở đâu?"
    ]

    result = extract_multiple_answers(context, questions, combine=False)

    if not result and len(context) < 100:
        return context.strip()

    return result if result else context.strip()


def process_cv_location(input_csv: str, output_csv: str = None) -> pd.DataFrame:
    """
    Process CV data to extract locations.
    """
    df = pd.read_csv(input_csv)

    locations = []
    for idx, row in df.iterrows():
        loc = extract_location(row.get('Workplace Desired', ''))
        locations.append({
            'UserID': row['UserID'],
            'location': loc
        })

    result_df = pd.DataFrame(locations)

    if output_csv:
        result_df.to_csv(output_csv, index=False)

    return result_df


if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent.parent
    input_path = base_dir / "huggingface + kaggle/dataset/kaggle/job-dataset-for-recommendation/USER_DATA_FINAL.csv"
    output_path = base_dir / "preprocessed/cv_location_temp.csv"

    result = process_cv_location(str(input_path), str(output_path))
