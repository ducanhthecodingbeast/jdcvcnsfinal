#!/usr/bin/env python3
"""
Main orchestrator for CV and JD information retrieval using embedding similarity.

This script:
1. Loads CV and JD data from preprocessed/ (cv1.csv, jd1.csv)
2. Extracts structured information from free-text fields using embedding-based retrieval
3. Combines all extracted fields into final cv1.csv and jd1.csv
4. Saves to /preprocessed/

Uses query variation + diversity-aware selection for better coverage.
"""

import pandas as pd
from pathlib import Path
import sys
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import embedding-based extraction functions (from retrieinfor1)
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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# PATH CONFIGURATION
# ============================================================================
# Inputs and outputs are in preprocessed/ as cv1.csv and jd1.csv
PREPROCESSED_DIR = PROJECT_ROOT / "preprocessed"

CV_INPUT = PREPROCESSED_DIR / "cv1.csv"
JD_INPUT = PREPROCESSED_DIR / "jd1.csv"

CV_OUTPUT = PREPROCESSED_DIR / "cv1.csv"
JD_OUTPUT = PREPROCESSED_DIR / "jd1.csv"


# ============================================================================
# CV PROCESSING
# ============================================================================
def process_all_cv(input_csv: Path) -> pd.DataFrame:
    """
    Process all CV records and extract structured information.

    Returns DataFrame with columns:
    UserID, summary, education_level, institution, major, skills,
    location, salary_expectation, desired_job
    """
    logger.info("Loading raw CV data...")
    df = pd.read_csv(input_csv)
    logger.info(f"Loaded {len(df)} CV records")

    results = []

    for idx, row in df.iterrows():
        if idx % 50 == 0:
            logger.info(f"Processing CV {idx}/{len(df)}")

        user_id = row['UserID']

        # Extract from each field using dedicated functions
        summary = extract_summary(row.get('Target', ''))

        edu = extract_education(row.get('Degree', ''))
        education_level = edu.get('education_level', '')
        institution = edu.get('institution', '')
        major = edu.get('major', '')

        skills = extract_skills(row.get('Skills', ''))

        location = extract_location(row.get('Workplace Desired', ''))

        salary_expectation = extract_salary(row.get('Desired Salary', ''))

        desired_job = extract_desired_job(row.get('Desired Job', ''))

        results.append({
            'UserID': user_id,
            'summary': summary,
            'education_level': education_level,
            'institution': institution,
            'major': major,
            'skills': skills,
            'location': location,
            'salary_expectation': salary_expectation,
            'desired_job': desired_job
        })

    result_df = pd.DataFrame(results)
    logger.info(f"CV processing complete: {len(result_df)} records")
    logger.info(f"  Non-empty summaries: {(result_df['summary'] != '').sum()}")
    logger.info(f"  Non-empty skills: {(result_df['skills'] != '').sum()}")

    return result_df


# ============================================================================
# JD PROCESSING
# ============================================================================
def process_all_jd(input_csv: Path) -> pd.DataFrame:
    """
    Process all JD records and extract structured information.

    Returns DataFrame with columns:
    JobID, job_description, responsibilities, requirements,
    required_education_level, required_major, required_skills,
    salary_offer, job_location
    """
    logger.info("Loading raw JD data...")
    df = pd.read_csv(input_csv)
    logger.info(f"Loaded {len(df)} JD records")

    results = []

    for idx, row in df.iterrows():
        if idx % 50 == 0:
            logger.info(f"Processing JD {idx}/{len(df)}")

        job_id = row['JobID']

        # Extract job description + responsibilities + requirements
        desc = extract_job_description(
            row.get('Job Description', ''),
            row.get('Job Requirements', '')
        )
        job_description = desc.get('job_description', '')
        responsibilities = desc.get('responsibilities', '')
        requirements = desc.get('requirements', '')

        # Extract education requirements
        edu_req = extract_education_requirements(
            row.get('Job Requirements', ''),
            row.get('Job Description', '')
        )
        required_education_level = edu_req.get('required_education_level', '')
        required_major = edu_req.get('required_major', '')

        # Extract required skills
        required_skills = extract_required_skills(
            row.get('Job Requirements', ''),
            row.get('Job Description', '')
        )

        # Extract salary offer
        salary_offer = extract_salary_offer(row.get('Salary', ''))

        # Extract job location
        job_location = extract_job_location(row.get('Job Address', ''))

        results.append({
            'JobID': job_id,
            'job_description': job_description,
            'responsibilities': responsibilities,
            'requirements': requirements,
            'required_education_level': required_education_level,
            'required_major': required_major,
            'required_skills': required_skills,
            'salary_offer': salary_offer,
            'job_location': job_location
        })

    result_df = pd.DataFrame(results)
    logger.info(f"JD processing complete: {len(result_df)} records")
    logger.info(f"  Non-empty descriptions: {(result_df['job_description'] != '').sum()}")
    logger.info(f"  Non-empty required skills: {(result_df['required_skills'] != '').sum()}")

    return result_df


# ============================================================================
# MAIN
# ============================================================================
def main():
    """
    Main entry point. Processes CV and JD data and saves to preprocessed/.

    DO NOT RUN until code has been reviewed.
    """
    logger.info("=" * 60)
    logger.info("Starting CV/JD Information Retrieval (Embedding Model + Diversity)")
    logger.info("=" * 60)

    # Ensure output directory exists
    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # --- Process CV ---
    logger.info("\n--- Processing CVs (from cv1.csv) ---")
    cv_df = process_all_cv(CV_INPUT)
    cv_df.to_csv(CV_OUTPUT, index=False)
    logger.info(f"Saved CV data to: {CV_OUTPUT}")

    # --- Process JD ---
    logger.info("\n--- Processing JDs (from jd1.csv) ---")
    jd_df = process_all_jd(JD_INPUT)
    jd_df.to_csv(JD_OUTPUT, index=False)
    logger.info(f"Saved JD data to: {JD_OUTPUT}")

    # --- Summary ---
    logger.info("\n" + "=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"CV output: {CV_OUTPUT} ({len(cv_df)} records)")
    logger.info(f"JD output: {JD_OUTPUT} ({len(jd_df)} records)")
    logger.info("\nCV columns: " + ", ".join(cv_df.columns.tolist()))
    logger.info("JD columns: " + ", ".join(jd_df.columns.tolist()))


if __name__ == "__main__":
    main()