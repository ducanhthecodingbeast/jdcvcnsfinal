"""
Extract location information from CV data using embedding similarity.
Source field: 'Workplace Desired'

Vietnamese queries.
"""

import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from embedding_model import extract_field_by_similarity


def extract_location(workplace_text: str) -> str:
    """
    Extract desired workplace location from CV using embedding similarity.

    Args:
        workplace_text: Raw 'Workplace Desired' field from user data (Vietnamese)

    Returns:
        Extracted location in Vietnamese
    """
    if pd.isna(workplace_text) or not str(workplace_text).strip():
        return ""

    context = str(workplace_text).strip()

    result = extract_field_by_similarity(context, "location")

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
