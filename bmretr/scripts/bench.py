#!/usr/bin/env python3
"""
Benchmark: compare word-level overlap (token probability) between
- retrieinfor (QA) output
- retrieinfor1 (embedding) output
against the gold standard perfect extraction in bmretr/bmcv.csv and bmjd.csv

Metrics per field:
- precision = |intersection| / |pred_tokens|
- recall    = |intersection| / |gold_tokens|
- f1

Also reports macro averages over records and fields.
"""

import pandas as pd
import re
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BMRETR = Path(__file__).parent.parent
GOLD_CV = BMRETR / "gold_mocked" / "cv.csv"
GOLD_JD = BMRETR / "gold_mocked" / "jd.csv"

QA_CV = BMRETR / "outputs" / "qa" / "cv.csv"
QA_JD = BMRETR / "outputs" / "qa" / "jd.csv"

EMB_CV = BMRETR / "outputs" / "emb" / "cv.csv"
EMB_JD = BMRETR / "outputs" / "emb" / "jd.csv"

# Fields to compare (must exist in gold + both outputs)
CV_FIELDS = [
    "summary",
    "education_level",
    "institution",
    "major",
    "skills",
    "location",
    "salary_expectation",
    "desired_job",
]

JD_FIELDS = [
    "job_description",
    "responsibilities",
    "requirements",
    "required_education_level",
    "required_major",
    "required_skills",
    "salary_offer",
    "job_location",
]


def tokenize(text: str) -> List[str]:
    if pd.isna(text) or not str(text).strip():
        return []
    # Lowercase, split on non-alphanumeric, keep Vietnamese letters
    t = str(text).lower()
    tokens = re.findall(r'[\wÀ-ỹ]+', t, flags=re.UNICODE)
    return tokens


def token_set(text: str) -> set:
    return set(tokenize(text))


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def precision_recall_f1(pred: set, gold: set) -> Tuple[float, float, float]:
    inter = len(pred & gold)
    if not pred:
        prec = 0.0
    else:
        prec = inter / len(pred)
    if not gold:
        rec = 0.0
    else:
        rec = inter / len(gold)
    if prec + rec == 0:
        f1 = 0.0
    else:
        f1 = 2 * prec * rec / (prec + rec)
    return prec, rec, f1


def compare_df(pred_df: pd.DataFrame, gold_df: pd.DataFrame, id_col: str, fields: List[str]) -> pd.DataFrame:
    """
    Return a long-form DataFrame with per-record per-field metrics.
    """
    rows = []
    for _, prow in pred_df.iterrows():
        rid = str(prow[id_col])
        gmask = gold_df[id_col].astype(str) == rid
        if not gmask.any():
            continue
        grow = gold_df[gmask].iloc[0]
        for f in fields:
            ptext = prow.get(f, "")
            gtext = grow.get(f, "")
            pset = token_set(ptext)
            gset = token_set(gtext)
            prec, rec, f1 = precision_recall_f1(pset, gset)
            rows.append({
                id_col: rid,
                "field": f,
                "pred_tokens": len(pset),
                "gold_tokens": len(gset),
                "intersection": len(pset & gset),
                "precision": prec,
                "recall": rec,
                "f1": f1,
            })
    return pd.DataFrame(rows)


def main():
    logger.info("Loading gold, QA, EMB outputs...")

    gold_cv = pd.read_csv(GOLD_CV)
    gold_jd = pd.read_csv(GOLD_JD)

    qa_cv = pd.read_csv(QA_CV) if QA_CV.exists() else pd.DataFrame()
    qa_jd = pd.read_csv(QA_JD) if QA_JD.exists() else pd.DataFrame()
    emb_cv = pd.read_csv(EMB_CV) if EMB_CV.exists() else pd.DataFrame()
    emb_jd = pd.read_csv(EMB_JD) if EMB_JD.exists() else pd.DataFrame()

    results = []

    if not qa_cv.empty:
        logger.info("Comparing QA (retrieinfor) vs Gold for CV")
        qa_cv_cmp = compare_df(qa_cv, gold_cv, "UserID", CV_FIELDS)
        qa_cv_cmp["system"] = "QA"
        qa_cv_cmp["type"] = "CV"
        results.append(qa_cv_cmp)
    else:
        logger.warning("QA CV output not found, skipping.")

    if not qa_jd.empty:
        logger.info("Comparing QA (retrieinfor) vs Gold for JD")
        qa_jd_cmp = compare_df(qa_jd, gold_jd, "JobID", JD_FIELDS)
        qa_jd_cmp["system"] = "QA"
        qa_jd_cmp["type"] = "JD"
        results.append(qa_jd_cmp)
    else:
        logger.warning("QA JD output not found, skipping.")

    if not emb_cv.empty:
        logger.info("Comparing EMB (retrieinfor1) vs Gold for CV")
        emb_cv_cmp = compare_df(emb_cv, gold_cv, "UserID", CV_FIELDS)
        emb_cv_cmp["system"] = "EMB"
        emb_cv_cmp["type"] = "CV"
        results.append(emb_cv_cmp)
    else:
        logger.warning("EMB CV output not found, skipping.")

    if not emb_jd.empty:
        logger.info("Comparing EMB (retrieinfor1) vs Gold for JD")
        emb_jd_cmp = compare_df(emb_jd, gold_jd, "JobID", JD_FIELDS)
        emb_jd_cmp["system"] = "EMB"
        emb_jd_cmp["type"] = "JD"
        results.append(emb_jd_cmp)
    else:
        logger.warning("EMB JD output not found, skipping.")

    if not results:
        logger.error("No results to compare. Run both retrievers first.")
        return

    all_cmp = pd.concat(results, ignore_index=True)

    # Summary
    summary = (
        all_cmp.groupby(["system", "type"])
        .agg(
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_f1=("f1", "mean"),
            n=("f1", "count"),
        )
        .reset_index()
    )

    print("\n" + "=" * 70)
    print("WORD-LEVEL OVERLAP BENCHMARK (vs gold perfect extraction)")
    print("=" * 70)
    print(all_cmp.to_string(index=False))
    print("\n--- SUMMARY (macro avg over records+fields) ---")
    print(summary.to_string(index=False))

    # Also save
    all_cmp.to_csv(BMRETR / "results" / "detail.csv", index=False)
    summary.to_csv(BMRETR / "results" / "summary.csv", index=False)
    print(f"\nSaved detail -> {BMRETR / 'results' / 'detail.csv'}")
    print(f"Saved summary -> {BMRETR / 'results' / 'summary.csv'}")


if __name__ == "__main__":
    main()