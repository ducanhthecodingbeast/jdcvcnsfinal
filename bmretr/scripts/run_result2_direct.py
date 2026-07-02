#!/usr/bin/env python3
"""
Direct runner for QA (retrieinfor) and EMB (retrieinfor1) on cv.md/jd.md.
Bypasses package imports by importing modules directly.
Outputs to bmretr/result/result2/ as QAcv.csv, QAjd.csv, EMBcv.csv, EMBjd.csv
"""

import sys
from pathlib import Path
import pandas as pd
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

BMRETR = ROOT / "bmretr"
RESULT_DIR = BMRETR / "result" / "result2"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CV_MD = BMRETR / "inputs" / "cv.md"
INPUT_JD_MD = BMRETR / "inputs" / "jd.md"

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


# ------------------------------
# Parse markdown inputs
# ------------------------------

def extract_section(text: str, headers: list[str]) -> str:
    next_headers = [
        "Mục tiêu nghề nghiệp", "Kỹ năng", "Học vấn",
        "Nơi làm việc mong muốn", "Mức lương mong muốn", "Công việc mong muốn",
        "Mô tả công việc", "Yêu cầu công việc", "Mức lương", "Địa điểm",
        "---"
    ]
    next_pat = "|".join(re.escape(h) for h in next_headers)

    for h in headers:
        idx = text.find(h)
        if idx == -1:
            continue
        start = idx + len(h)
        rest = text[start:]
        m = re.search(rf"\n(?={next_pat})", rest)
        content = rest[:m.start()].strip() if m else rest.strip()
        content = re.sub(r"^[:：\-\s]+", "", content).strip()
        if content:
            return content
    return ""


def parse_cv_md(md_path: Path) -> pd.DataFrame:
    text = md_path.read_text(encoding="utf-8")
    records = re.split(r'\n---\n|(?=CV \d+ – UserID)', text)

    rows = []
    for block in records:
        if not block.strip() or "CV Input Data" in block:
            continue
        uid_match = re.search(r'UserID\s+(\d+)', block)
        if not uid_match:
            continue
        user_id = int(uid_match.group(1))

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
    text = md_path.read_text(encoding="utf-8")
    records = re.split(r'\n---\n|(?=JD \d+ – )', text)

    rows = []
    for block in records:
        if not block.strip() or "JD Input Data" in block:
            continue
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


# ------------------------------
# QA (retrieinfor) processing
# ------------------------------

def run_qa(cv_df: pd.DataFrame, jd_df: pd.DataFrame):
    logger.info("=== Running retrieinfor (QA) ===")

    from retrieinfor.cv.summary import extract_summary
    from retrieinfor.cv.education import extract_education
    from retrieinfor.cv.skillspair import extract_skills
    from retrieinfor.cv.salary import extract_salary
    from retrieinfor.cv.location import extract_location
    from retrieinfor.cv.desiredjob import extract_desired_job

    from retrieinfor.jd.jobdesciption import extract_job_description
    from retrieinfor.jd.education import extract_education_requirements
    from retrieinfor.jd.skillspair import extract_required_skills
    from retrieinfor.jd.salaryoffer import extract_salary_offer
    from retrieinfor.jd.location import extract_job_location

    # CV
    qa_cv_rows = []
    for _, row in cv_df.iterrows():
        uid = row['UserID']
        summary = extract_summary(row.get('Target', ''))

        edu = extract_education(row.get('Degree', ''))
        education_level = edu.get('education_level', '')
        institution = edu.get('institution', '')
        major = edu.get('major', '')

        skills = extract_skills(row.get('Skills', ''))
        location = extract_location(row.get('Workplace Desired', ''))
        salary_expectation = extract_salary(row.get('Desired Salary', ''))
        desired_job = extract_desired_job(row.get('Desired Job', ''))

        qa_cv_rows.append({
            'UserID': uid,
            'summary': summary,
            'education_level': education_level,
            'institution': institution,
            'major': major,
            'skills': skills,
            'location': location,
            'salary_expectation': salary_expectation,
            'desired_job': desired_job
        })

    qa_cv = pd.DataFrame(qa_cv_rows)
    qa_cv.to_csv(QA_CV_OUT, index=False)
    logger.info(f"QA CV saved to {QA_CV_OUT}")

    # JD
    qa_jd_rows = []
    for _, row in jd_df.iterrows():
        jid = row['JobID']

        desc = extract_job_description(
            row.get('Job Description', ''),
            row.get('Job Requirements', '')
        )
        job_description = desc.get('job_description', '')
        responsibilities = desc.get('responsibilities', '')
        requirements = desc.get('requirements', '')

        edu_req = extract_education_requirements(
            row.get('Job Requirements', ''),
            row.get('Job Description', '')
        )
        required_education_level = edu_req.get('required_education_level', '')
        required_major = edu_req.get('required_major', '')

        required_skills = extract_required_skills(
            row.get('Job Requirements', ''),
            row.get('Job Description', '')
        )

        salary_offer = extract_salary_offer(row.get('Salary', ''))
        job_location = extract_job_location(row.get('Job Address', ''))

        qa_jd_rows.append({
            'JobID': jid,
            'job_description': job_description,
            'responsibilities': responsibilities,
            'requirements': requirements,
            'required_education_level': required_education_level,
            'required_major': required_major,
            'required_skills': required_skills,
            'salary_offer': salary_offer,
            'job_location': job_location
        })

    qa_jd = pd.DataFrame(qa_jd_rows)
    qa_jd.to_csv(QA_JD_OUT, index=False)
    logger.info(f"QA JD saved to {QA_JD_OUT}")

    return qa_cv, qa_jd


# ------------------------------
# EMB (retrieinfor1) processing
# ------------------------------

def run_emb(cv_df: pd.DataFrame, jd_df: pd.DataFrame):
    logger.info("=== Running retrieinfor1 (EMB) ===")

    from retrieinfor1.cv.summary import extract_summary
    from retrieinfor1.cv.education import extract_education
    from retrieinfor1.cv.skillspair import extract_skills
    from retrieinfor1.cv.salary import extract_salary
    from retrieinfor1.cv.location import extract_location
    from retrieinfor1.cv.desiredjob import extract_desired_job

    from retrieinfor1.jd.jobdesciption import extract_job_description
    from retrieinfor1.jd.education import extract_education_requirements
    from retrieinfor1.jd.skillspair import extract_required_skills
    from retrieinfor1.jd.salaryoffer import extract_salary_offer
    from retrieinfor1.jd.location import extract_job_location

    # CV
    emb_cv_rows = []
    for _, row in cv_df.iterrows():
        uid = row['UserID']
        summary = extract_summary(row.get('Target', ''))

        edu = extract_education(row.get('Degree', ''))
        education_level = edu.get('education_level', '')
        institution = edu.get('institution', '')
        major = edu.get('major', '')

        skills = extract_skills(row.get('Skills', ''))
        location = extract_location(row.get('Workplace Desired', ''))
        salary_expectation = extract_salary(row.get('Desired Salary', ''))
        desired_job = extract_desired_job(row.get('Desired Job', ''))

        emb_cv_rows.append({
            'UserID': uid,
            'summary': summary,
            'education_level': education_level,
            'institution': institution,
            'major': major,
            'skills': skills,
            'location': location,
            'salary_expectation': salary_expectation,
            'desired_job': desired_job
        })

    emb_cv = pd.DataFrame(emb_cv_rows)
    emb_cv.to_csv(EMB_CV_OUT, index=False)
    logger.info(f"EMB CV saved to {EMB_CV_OUT}")

    # JD
    emb_jd_rows = []
    for _, row in jd_df.iterrows():
        jid = row['JobID']

        desc = extract_job_description(
            row.get('Job Description', ''),
            row.get('Job Requirements', '')
        )
        job_description = desc.get('job_description', '')
        responsibilities = desc.get('responsibilities', '')
        requirements = desc.get('requirements', '')

        edu_req = extract_education_requirements(
            row.get('Job Requirements', ''),
            row.get('Job Description', '')
        )
        required_education_level = edu_req.get('required_education_level', '')
        required_major = edu_req.get('required_major', '')

        required_skills = extract_required_skills(
            row.get('Job Requirements', ''),
            row.get('Job Description', '')
        )

        salary_offer = extract_salary_offer(row.get('Salary', ''))
        job_location = extract_job_location(row.get('Job Address', ''))

        emb_jd_rows.append({
            'JobID': jid,
            'job_description': job_description,
            'responsibilities': responsibilities,
            'requirements': requirements,
            'required_education_level': required_education_level,
            'required_major': required_major,
            'required_skills': required_skills,
            'salary_offer': salary_offer,
            'job_location': job_location
        })

    emb_jd = pd.DataFrame(emb_jd_rows)
    emb_jd.to_csv(EMB_JD_OUT, index=False)
    logger.info(f"EMB JD saved to {EMB_JD_OUT}")

    return emb_cv, emb_jd


# ------------------------------
# Main
# ------------------------------

def main():
    logger.info("=" * 60)
    logger.info("Direct run: QA + EMB on cv.md/jd.md → result/result2")
    logger.info("=" * 60)

    # Parse
    logger.info("Parsing markdown inputs...")
    cv_df = parse_cv_md(INPUT_CV_MD)
    jd_df = parse_jd_md(INPUT_JD_MD)
    cv_df.to_csv(TMP_CV_CSV, index=False)
    jd_df.to_csv(TMP_JD_CSV, index=False)

    # QA
    qa_cv, qa_jd = run_qa(cv_df, jd_df)

    # EMB
    emb_cv, emb_jd = run_emb(cv_df, jd_df)

    # Copy to final locations
    import shutil
    shutil.copy(QA_CV_OUT, FINAL_QA_CV)
    shutil.copy(QA_JD_OUT, FINAL_QA_JD)
    shutil.copy(EMB_CV_OUT, FINAL_EMB_CV)
    shutil.copy(EMB_JD_OUT, FINAL_EMB_JD)

    logger.info("\n" + "=" * 60)
    logger.info("DONE. Outputs in result/result2/:")
    logger.info(f"  QA:  {FINAL_QA_CV.name}, {FINAL_QA_JD.name}")
    logger.info(f"  EMB: {FINAL_EMB_CV.name}, {FINAL_EMB_JD.name}")
    logger.info("=" * 60)

    print("\nQA CV columns:", list(qa_cv.columns))
    print("QA JD columns:", list(qa_jd.columns))
    print("EMB CV columns:", list(emb_cv.columns))
    print("EMB JD columns:", list(emb_jd.columns))


if __name__ == "__main__":
    main()