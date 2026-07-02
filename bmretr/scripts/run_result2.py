#!/usr/bin/env python3
"""
Run QA (retrieinfor) and EMB (retrieinfor1) on cv.md and jd.md inputs.
Produces outputs in bmretr/result/result2/ as:
  QAcv.csv, QAjd.csv (from retrieinfor/QA)
  EMBcv.csv, EMBjd.csv (from retrieinfor1/EMB)

Uses gold_mocked as the reference structure.
"""

import sys
from pathlib import Path
import pandas as pd
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent.parent
BMRETR = ROOT / "bmretr"
RESULT_DIR = BMRETR / "result" / "result2"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CV_MD = BMRETR / "inputs" / "cv.md"
INPUT_JD_MD = BMRETR / "inputs" / "jd.md"

# Temporary working CSVs (parsed from md)
TMP_DIR = BMRETR / "tmp_result2"
TMP_DIR.mkdir(exist_ok=True)
TMP_CV_CSV = TMP_DIR / "cv_input.csv"
TMP_JD_CSV = TMP_DIR / "jd_input.csv"

# QA outputs (retrieinfor)
QA_OUT_DIR = BMRETR / "outputs" / "qa_result2"
QA_OUT_DIR.mkdir(parents=True, exist_ok=True)
QA_CV_OUT = QA_OUT_DIR / "cv.csv"
QA_JD_OUT = QA_OUT_DIR / "jd.csv"

# EMB outputs (retrieinfor1)
EMB_OUT_DIR = BMRETR / "outputs" / "emb_result2"
EMB_OUT_DIR.mkdir(parents=True, exist_ok=True)
EMB_CV_OUT = EMB_OUT_DIR / "cv.csv"
EMB_JD_OUT = EMB_OUT_DIR / "jd.csv"

# Final result locations
FINAL_QA_CV = RESULT_DIR / "QAcv.csv"
FINAL_QA_JD = RESULT_DIR / "QAjd.csv"
FINAL_EMB_CV = RESULT_DIR / "EMBcv.csv"
FINAL_EMB_JD = RESULT_DIR / "EMBjd.csv"


def parse_cv_md(md_path: Path) -> pd.DataFrame:
    """
    Parse cv.md into DataFrame matching USER_DATA_FINAL.csv structure.
    Columns needed: UserID, Target, Skills, Degree, Workplace Desired, Desired Salary, Desired Job
    """
    text = md_path.read_text(encoding="utf-8")

    # Split by CV record separators
    # Pattern: "CV N – UserID XXXXX (title)" or "---"
    records = re.split(r'\n---\n|(?=CV \d+ – UserID)', text)

    rows = []
    for block in records:
        if not block.strip() or "CV Input Data" in block:
            continue

        # Extract UserID
        uid_match = re.search(r'UserID\s+(\d+)', block)
        if not uid_match:
            continue
        user_id = int(uid_match.group(1))

        # Extract sections
        target = extract_section(block, ["Mục tiêu nghề nghiệp:", "Mục tiêu nghề nghiệp"])
        skills = extract_section(block, ["Kỹ năng:", "Kỹ năng"])
        degree = extract_section(block, ["Học vấn:", "Học vấn"])
        workplace = extract_section(block, ["Nơi làm việc mong muốn:", "Nơi làm việc mong muốn"])
        salary = extract_section(block, ["Mức lương mong muốn:", "Mức lương mong muốn"])
        desired = extract_section(block, ["Công việc mong muốn:", "Công việc mong muốn"])

        rows.append({
            "UserID": user_id,
            "Target": target,
            "Skills": skills,
            "Degree": degree,
            "Workplace Desired": workplace,
            "Desired Salary": salary,
            "Desired Job": desired,
        })

    df = pd.DataFrame(rows)
    logger.info(f"Parsed {len(df)} CV records from {md_path}")
    return df


def parse_jd_md(md_path: Path) -> pd.DataFrame:
    """
    Parse jd.md into DataFrame matching JOB_DATA_FINAL.csv structure.
    Columns needed: JobID, Job Description, Job Requirements, Salary, Job Address
    """
    text = md_path.read_text(encoding="utf-8")

    records = re.split(r'\n---\n|(?=JD \d+ – )', text)

    rows = []
    for block in records:
        if not block.strip() or "JD Input Data" in block:
            continue

        # Extract JobID
        jid_match = re.search(r'JD\s+(\d+)', block)
        if not jid_match:
            continue
        job_id = int(jid_match.group(1))

        desc = extract_section(block, ["Mô tả công việc:", "Mô tả công việc"])
        reqs = extract_section(block, ["Yêu cầu công việc:", "Yêu cầu công việc"])
        salary = extract_section(block, ["Mức lương:", "Mức lương"])
        address = extract_section(block, ["Địa điểm:", "Địa điểm"])

        rows.append({
            "JobID": job_id,
            "Job Description": desc,
            "Job Requirements": reqs,
            "Salary": salary,
            "Job Address": address,
        })

    df = pd.DataFrame(rows)
    logger.info(f"Parsed {len(df)} JD records from {md_path}")
    return df


def extract_section(text: str, headers: list[str]) -> str:
    """
    Extract text after a header until next known header or end.
    """
    # Build pattern for all possible next headers we care about
    next_headers = [
        "Mục tiêu nghề nghiệp", "Kỹ năng", "Học vấn",
        "Nơi làm việc mong muốn", "Mức lương mong muốn", "Công việc mong muốn",
        "Mô tả công việc", "Yêu cầu công việc", "Mức lương", "Địa điểm",
        "---"
    ]
    # Escape for regex
    next_pat = "|".join(re.escape(h) for h in next_headers)

    for h in headers:
        # Find start after header
        idx = text.find(h)
        if idx == -1:
            continue
        start = idx + len(h)
        # Capture until next header or end
        rest = text[start:]
        # Stop at next header occurrence
        m = re.search(rf"\n(?={next_pat})", rest)
        if m:
            content = rest[:m.start()].strip()
        else:
            content = rest.strip()
        # Clean leading separators like ": " or "–"
        content = re.sub(r"^[:：\-\s]+", "", content).strip()
        if content:
            return content
    return ""


def run_qa(cv_csv: Path, jd_csv: Path):
    """
    Run retrieinfor (QA) on the given CSVs.
    Monkey-patch paths and call processing functions.
    """
    logger.info("=== Running retrieinfor (QA) ===")

    import retrieinfor.main as qa_main
    from retrieinfor.main import process_all_cv, process_all_jd

    # Patch paths
    qa_main.CV_RAW = cv_csv
    qa_main.JD_RAW = jd_csv
    qa_main.CV_OUTPUT = QA_CV_OUT
    qa_main.JD_OUTPUT = QA_JD_OUT

    cv_df = process_all_cv(cv_csv)
    cv_df.to_csv(QA_CV_OUT, index=False)
    logger.info(f"QA CV saved to {QA_CV_OUT}")

    jd_df = process_all_jd(jd_csv)
    jd_df.to_csv(QA_JD_OUT, index=False)
    logger.info(f"QA JD saved to {QA_JD_OUT}")

    return cv_df, jd_df


def run_emb(cv_csv: Path, jd_csv: Path):
    """
    Run retrieinfor1 (EMB) on the given CSVs.
    """
    logger.info("=== Running retrieinfor1 (EMB) ===")

    import retrieinfor1.main as emb_main
    from retrieinfor1.main import process_all_cv, process_all_jd

    emb_main.CV_INPUT = cv_csv
    emb_main.JD_INPUT = jd_csv
    emb_main.CV_OUTPUT = EMB_CV_OUT
    emb_main.JD_OUTPUT = EMB_JD_OUT

    cv_df = process_all_cv(cv_csv)
    cv_df.to_csv(EMB_CV_OUT, index=False)
    logger.info(f"EMB CV saved to {EMB_CV_OUT}")

    jd_df = process_all_jd(jd_csv)
    jd_df.to_csv(EMB_JD_OUT, index=False)
    logger.info(f"EMB JD saved to {EMB_JD_OUT}")

    return cv_df, jd_df


def main():
    logger.info("=" * 60)
    logger.info("Running QA and EMB on cv.md / jd.md → result/result2")
    logger.info("=" * 60)

    # 1) Parse markdown inputs → CSVs
    logger.info("Parsing markdown inputs...")
    cv_df = parse_cv_md(INPUT_CV_MD)
    jd_df = parse_jd_md(INPUT_JD_MD)

    cv_df.to_csv(TMP_CV_CSV, index=False)
    jd_df.to_csv(TMP_JD_CSV, index=False)
    logger.info(f"Temporary CSVs written: {TMP_CV_CSV}, {TMP_JD_CSV}")

    # 2) Run QA
    qa_cv, qa_jd = run_qa(TMP_CV_CSV, TMP_JD_CSV)

    # 3) Run EMB
    emb_cv, emb_jd = run_emb(TMP_CV_CSV, TMP_JD_CSV)

    # 4) Copy to final result locations with requested names
    import shutil
    shutil.copy(QA_CV_OUT, FINAL_QA_CV)
    shutil.copy(QA_JD_OUT, FINAL_QA_JD)
    shutil.copy(EMB_CV_OUT, FINAL_EMB_CV)
    shutil.copy(EMB_JD_OUT, FINAL_EMB_JD)

    logger.info("\n" + "=" * 60)
    logger.info("DONE. Final outputs in result/result2/:")
    logger.info(f"  QA:  {FINAL_QA_CV.name}, {FINAL_QA_JD.name}")
    logger.info(f"  EMB: {FINAL_EMB_CV.name}, {FINAL_EMB_JD.name}")
    logger.info("=" * 60)

    # Quick sanity print
    print("\nQA CV columns:", list(qa_cv.columns))
    print("QA JD columns:", list(qa_jd.columns))
    print("EMB CV columns:", list(emb_cv.columns))
    print("EMB JD columns:", list(emb_jd.columns))


if __name__ == "__main__":
    main()