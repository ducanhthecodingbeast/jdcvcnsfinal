import csv
from pathlib import Path

src = Path("/home/aiface/jdcvcnsfinal/huggingface + kaggle/dataset/kaggle/job-dataset-for-recommendation")
out = Path("/home/aiface/jdcvcnsfinal/phase1")

for csvf in src.glob("*.csv"):
    rows = list(csv.reader(open(csvf)))
    md = "| " + " | ".join(rows[0]) + " |\n" + "|---" * len(rows[0]) + "|\n" + "\n".join("| " + " | ".join(r) + " |" for r in rows[1:])
    open(out/(csvf.stem+".md"), "w").write(md)