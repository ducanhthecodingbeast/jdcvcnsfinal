#!/usr/bin/env python3
"""
Run both retrieinfor (QA) and retrieinfor1 (embedding) on the first 10 CV/JD,
producing bmretr/retrieinfor_out/cv.csv and bmretr/retrieinfor1_out/cv.csv
(and same for jd).

This assumes the two packages are importable as:
  import retrieinfor.main as qa_main
  import retrieinfor1.main as emb_main

They will be called with overridden paths via monkey-patching the constants.
"""

import sys
from pathlib import Path
import pandas as pd
import logging

# Ensure we can import retrieinfor and retrieinfor1 as top-level packages
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
BMRETR = ROOT / "bmretr"
INPUTS = BMRETR / "raw"
QA_OUT = BMRETR / "outputs" / "qa"
EMB_OUT = BMRETR / "outputs" / "emb"
QA_OUT.mkdir(parents=True, exist_ok=True)
EMB_OUT.mkdir(parents=True, exist_ok=True)

CV_IN = INPUTS / "cv_first10.csv"
JD_IN = INPUTS / "jd_first10.csv"

def run_retrieinfor():
    """
    Use the QA-based retrieinfor (original) to process first 10.
    We temporarily override its path constants and call the processing functions.
    """
    logger.info("=== Running retrieinfor (QA) on first 10 ===")

    # We will import the modules and monkey-patch their globals
    import retrieinfor.main as qa_main
    from retrieinfor.main import process_all_cv, process_all_jd

    # Patch paths
    qa_main.CV_RAW = CV_IN
    qa_main.JD_RAW = JD_IN
    qa_main.CV_OUTPUT = QA_OUT / "cv.csv"
    qa_main.JD_OUTPUT = QA_OUT / "jd.csv"

    cv_df = process_all_cv(CV_IN)
    cv_df.to_csv(QA_OUT / "cv.csv", index=False)
    logger.info(f"QA CV saved to {QA_OUT / 'cv.csv'}")

    jd_df = process_all_jd(JD_IN)
    jd_df.to_csv(QA_OUT / "jd.csv", index=False)
    logger.info(f"QA JD saved to {QA_OUT / 'jd.csv'}")


def run_retrieinfor1():
    """
    Use the embedding-based retrieinfor1 on first 10.
    """
    logger.info("=== Running retrieinfor1 (embedding) on first 10 ===")

    import retrieinfor1.main as emb_main
    from retrieinfor1.main import process_all_cv, process_all_jd

    emb_main.CV_INPUT = CV_IN
    emb_main.JD_INPUT = JD_IN
    emb_main.CV_OUTPUT = EMB_OUT / "cv.csv"
    emb_main.JD_OUTPUT = EMB_OUT / "jd.csv"

    cv_df = process_all_cv(CV_IN)
    cv_df.to_csv(EMB_OUT / "cv.csv", index=False)
    logger.info(f"EMB CV saved to {EMB_OUT / 'cv.csv'}")

    jd_df = process_all_jd(JD_IN)
    jd_df.to_csv(EMB_OUT / "jd.csv", index=False)
    logger.info(f"EMB JD saved to {EMB_OUT / 'jd.csv'}")


if __name__ == "__main__":
    run_retrieinfor()
    run_retrieinfor1()
    print("\nDone. Outputs:")
    print(f"  QA:   {QA_OUT}/cv.csv  {QA_OUT}/jd.csv")
    print(f"  EMB:  {EMB_OUT}/cv.csv {EMB_OUT}/jd.csv")