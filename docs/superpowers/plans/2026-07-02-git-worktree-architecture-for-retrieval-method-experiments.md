# Git Worktree Architecture for CV/JD Retrieval Method Experiments

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Design and document an architecture for using git worktrees to isolate and benchmark different CV/JD information retrieval methods against a shared gold standard, enabling reproducible experiments without cross-contamination.

**Architecture:** Worktrees provide isolated branches for each retrieval approach (QA, EMB, structured pipeline, ensemble, etc.). Each worktree contains method-specific extraction code while sharing immutable benchmark infrastructure (gold standards, inputs, scripts). A comparison harness aggregates results across worktrees. Superpowers skills orchestrate the lifecycle: worktree creation, TDD implementation, benchmarking, and result merging.

**Tech Stack:** Python 3, pandas, git worktrees, existing retrieinfor/retrieinfor1 codebases, bmretr/ benchmarking infrastructure

---

## Global Constraints

- Primary language is Python; all new code must be Python
- Must work with existing `bmretr/` benchmarking infrastructure without duplicating gold/inputs
- Worktree directories MUST be git-ignored (`.worktrees/` or similar)
- Follow superpowers philosophy: YAGNI, surgical changes, TDD, frequent commits
- Each worktree must produce comparable output (CSV with standard schema) for benchmarking
- Do NOT modify `bmretr/gold/`, `bmretr/gold_mocked/`, `bmretr/raw/`, or `bmretr/scripts/` from within worktrees
- Benchmark scripts run from project root or `bmretr/`, never from within a worktree
- Worktree contents are ephemeral until explicitly merged; results must be committed to main for persistence

---

## Context and Current State

### Existing Architecture

**Two parallel retrieval implementations exist:**
- `retrieinfor/` — QA-based extraction using `timpal0l/mdeberta-v3-base-squad2`
  - Vietnamese questions → extractive QA → answer spans
  - Files: `main.py`, `qa_model.py`, `cv/*.py`, `jd/*.py`
- `retrieinfor1/` — Embedding-based extraction using `BAAI/bge-m3`
  - Query variants → semantic similarity → cosine → top-k chunks
  - Files: `main.py`, `embedding_model.py`, `cv/*.py`, `jd/*.py`

**Both extract identical field schemas:**
- CV: `summary`, `education_level`, `institution`, `major`, `skills`, `location`, `salary_expectation`, `desired_job`
- JD: `job_description`, `responsibilities`, `requirements`, `required_education_level`, `required_major`, `required_skills`, `salary_offer`, `job_location`

**Benchmarking infrastructure (`bmretr/`):**
- Gold standard: `gold/cv.csv`, `gold/jd.csv` (human-curated, 10 records)
- Mocked gold: `gold_mocked/` (fills empty edu/major for fair comparison)
- Raw inputs: `raw/cv_first10.csv`, `raw/jd_first10.csv`
- Scripts:
  - `run_result2.py` — parses markdown inputs, monkey-patches paths, runs both retrievers
  - `bench.py` — token-overlap metrics (precision/recall/F1 via word intersection)
- Results: `result/result1/`, `result/result2/` with `detail.csv`, `summary.csv`, `best.csv`, `wins.csv`

**Current performance (result2):**
- CV: EMB 0.62 F1, QA 0.56 F1
- JD: QA 0.47 F1, EMB 0.38 F1

### Critical Analysis (from PATTERN_ANALYSIS_AND_SOLUTIONS.md)

Systematic problems preventing 0.9-1.0 precision:
1. **Context boundary failures** — over-extraction (5-30x token ratios)
2. **Semantic vs literal mismatch** — correct meaning, wrong tokens
3. **Missing content** — zero extraction when information is implicit
4. **Noise inclusion** — irrelevant surrounding text
5. **Field confusion** — extracting from wrong document sections

Proposed solutions (not yet implemented):
- Structured extraction pipeline (section identification → routing → field rules)
- Multi-stage refinement (coarse → boundary trim → validate → normalize)
- Field-specific heuristics/validators
- Ensemble with verification
- Hierarchical context windows
- Post-extraction canonicalization

### Why Git Worktrees?

**Problem:** Testing new retrieval methods requires modifying `retrieinfor/` or `retrieinfor1/`. This:
- Risks polluting working code
- Makes it hard to compare methods side-by-side
- Prevents running experiments in parallel
- Complicates rollback if an approach fails

**Solution:** Each worktree is an isolated branch with its own retrieval implementation. All worktrees share the same `bmretr/` gold standard and inputs. Results are comparable because outputs conform to the same schema.

---

## Architecture Overview

### Worktree Structure

```
project-root/
├── .worktrees/                          # Git-ignored (add to .gitignore)
│   ├── retrieval-qa-baseline/           # Current QA implementation (baseline)
│   │   ├── retrieinfor/                 # QA code (copied or symlinked from main)
│   │   ├── bmretr/                      # NOT here — shared from root
│   │   └── README.md                    # Documents what this tests
│   │
│   ├── retrieval-emb-baseline/          # Current EMB implementation (baseline)
│   │   ├── retrieinfor1/                # EMB code (copied or symlinked from main)
│   │   ├── bmretr/                      # NOT here — shared from root
│   │   └── README.md
│   │
│   ├── retrieval-structured-pipeline/   # Proposed: section-aware extraction
│   │   ├── retrieinfor_structured/      # New module (NOT overwriting existing)
│   │   ├── bmretr/                      # NOT here — shared from root
│   │   └── README.md
│   │
│   ├── retrieval-ensemble/              # Proposed: QA + EMB cross-validation
│   │   ├── retrieinfor_ensemble/        # New module
│   │   ├── bmretr/                      # NOT here — shared from root
│   │   └── README.md
│   │
│   └── retrieval-hierarchical/          # Proposed: hierarchical context windows
│       ├── retrieinfor_hierarchical/    # New module
│       ├── bmretr/                      # NOT here — shared from root
│       └── README.md
│
├── bmretr/                              # SHARED — never modified from worktrees
│   ├── gold/
│   ├── gold_mocked/
│   ├── raw/
│   ├── scripts/
│   │   ├── run_result2.py               # Runs retrieval methods (monkey-patches)
│   │   └── bench.py                     # Token-overlap metrics
│   └── result/
│       └── result2/
│           └── ... (generated outputs)
│
├── retrieinfor/                         # On main branch (baseline reference)
├── retrieinfor1/                        # On main branch (baseline reference)
├── .gitignore                           # Must include .worktrees/
└── docs/
    └── superpowers/
        └── plans/
            └── 2026-07-02-git-worktree-architecture-for-retrieval-method-experiments.md  # This plan
```

### Key Design Decisions

**1. Worktree directory location:** `.worktrees/` (hidden, git-ignored, project-local)

**2. Method naming convention:** `retrieval-{method-name}/`
- `qa-baseline` — existing QA (for reference)
- `emb-baseline` — existing EMB (for reference)
- `structured-pipeline` — section-aware extraction
- `ensemble` — cross-validation of multiple methods
- `hierarchical` — hierarchical context windows

**3. Code organization within worktree:**
- Option A (preferred for surgical changes): New directory per method (e.g., `retrieinfor_structured/`)
- Option B (not recommended): Overwrite `retrieinfor/` inside worktree (risky, blurs identity)

We choose Option A: each worktree introduces a new module directory. This:
- Preserves baseline code for comparison
- Makes it explicit which method is being tested
- Avoids accidental overwrites

**4. Shared benchmark infrastructure:**
- `bmretr/` lives ONLY at project root
- Worktrees do NOT contain copies of `bmretr/`
- Benchmark scripts run from root, monkey-patching to point at worktree's retrieval module

**5. Output schema contract:**
- Every retrieval method MUST produce CSV with exact columns:
  - CV: `UserID,summary,education_level,institution,major,skills,location,salary_expectation,desired_job`
  - JD: `JobID,job_description,responsibilities,requirements,required_education_level,required_major,required_skills,salary_offer,job_location`
- This ensures `bench.py` can compare any method against gold without modification

---

## Component Responsibilities

### 1. Worktree Manager (Human + Superpowers Skills)

**Responsibilities:**
- Create worktree via `superpowers:using-git-worktrees`
- Name worktree according to convention
- Document worktree purpose in `README.md`
- Clean up abandoned worktrees

**Does NOT:**
- Modify shared `bmretr/` infrastructure
- Run benchmarks from inside worktree

### 2. Retrieval Method Implementer (Per Worktree)

**Responsibilities:**
- Implement extraction logic in worktree-specific module
- Follow TDD: write failing test → minimal code → pass → commit
- Ensure output CSV conforms to schema contract
- Write worktree-local README documenting the approach
- Commit results (CSV outputs) to worktree branch

**Interfaces:**
- Consumes: raw CV/JD text (via monkey-patched paths)
- Produces: CSV with standard schema (written to worktree-local output dir)

**Example module structure (for `retrieval-structured-pipeline`):**
```
retrieinfor_structured/
├── main.py                    # Orchestrator: process_all_cv, process_all_jd
├── section_parser.py          # Identify document sections (headers, lists)
├── router.py                  # Route field to appropriate section
├── validators.py              # Field-specific validation rules
├── normalizers.py             # Canonicalization rules
├── cv/
│   ├── education.py           # Structured education extraction
│   └── ...
└── jd/
    ├── education.py
    └── ...
```

### 3. Benchmark Harness (`bmretr/scripts/`)

**Responsibilities:**
- `run_result2.py` — extended to accept worktree method path
  - Currently hardcodes `retrieinfor` and `retrieinfor1`
  - Must be generalized to accept arbitrary method module path
- `bench.py` — unchanged (already generic)
  - Reads gold from fixed paths
  - Reads prediction from configurable paths
  - Outputs detail/summary CSVs

**Current limitation:** `run_result2.py` hardcodes two methods. We need to extend it to:
- Accept command-line args for method module paths
- Or, read a config file listing active methods

**Proposed extension (minimal change):**
```bash
# Run specific method from worktree
python bmretr/scripts/run_result2.py \
  --method structured \
  --module-path .worktrees/retrieval-structured-pipeline/retrieinfor_structured \
  --output-dir bmretr/outputs/structured_result2
```

### 4. Result Aggregator (New Component)

**Location:** `bmretr/scripts/aggregate_results.py` (new file)

**Responsibilities:**
- Scan `bmretr/result/*/ ` directories for method outputs
- Run `bench.py` for each method against gold
- Aggregate into cross-method comparison table
- Identify best method per field
- Generate summary report

**Example output (`bmretr/result/comparison/cross_method_summary.csv`):**
```
method,field,mean_precision,mean_recall,mean_f1
qa-baseline,summary,0.XXX,0.XXX,0.XXX
emb-baseline,summary,0.XXX,0.XXX,0.XXX
structured-pipeline,summary,0.XXX,0.XXX,0.XXX
ensemble,summary,0.XXX,0.XXX,0.XXX
```

### 5. Superpowers Workflow Integration

**Using Git Worktrees (`superpowers:using-git-worktrees`):**
- Ensures `.worktrees/` is created and ignored
- Verifies isolation before implementation
- Each retrieval experiment starts with: "I'm using using-git-worktrees skill"

**Writing Plans (`superpowers:writing-plans`):**
- Each new retrieval method gets its own implementation plan
- Plan references this architecture document as context
- Plan decomposes method implementation into TDD tasks

**Subagent-Driven Development (`superpowers:subagent-driven-development`):**
- Per-task implementer subagent works inside worktree
- Task reviewer verifies: (1) spec compliance, (2) output schema contract
- Final branch review ensures results are committed and comparable

**Test-Driven Development (`superpowers:test-driven-development`):**
- Every retrieval function has a test
- Tests use small synthetic inputs (not full 10-record benchmark)
- Tests verify schema contract (output has required columns, non-empty for known inputs)

**Executing Plans (`superpowers:executing-plans`):**
- Alternative to subagent-driven for simpler methods
- Still requires worktree isolation first

**Brainstorming (`superpowers:brainstorming`):**
- Before designing a new retrieval method, brainstorm approach
- Document design in `docs/superpowers/specs/YYYY-MM-DD-<method>-design.md`
- Only after design approval, invoke writing-plans

---

## Data Flow

### Single Worktree Experiment Flow

```
1. Create worktree
   git worktree add .worktrees/retrieval-structured-pipeline -b retrieval-structured-pipeline

2. Implement method (inside worktree)
   - Write failing tests
   - Implement extraction logic
   - Ensure output schema matches contract
   - Commit code + tests

3. Run benchmark (from project root)
   python bmretr/scripts/run_result2.py \
     --method structured \
     --module-path .worktrees/retrieval-structured-pipeline/retrieinfor_structured \
     --output-dir bmretr/outputs/structured_result2

4. Compute metrics (from project root)
   python bmretr/scripts/bench.py \
     --pred-cv bmretr/outputs/structured_result2/cv.csv \
     --pred-jd bmretr/outputs/structured_result2/jd.csv \
     --gold-cv bmretr/gold_mocked/cv.csv \
     --gold-jd bmretr/gold_mocked/jd.csv \
     --out-dir bmretr/result/structured_result2

5. Commit results (inside worktree)
   git add bmretr/result/structured_result2/
   git commit -m "bench: structured-pipeline results on 10-record gold"

6. (Optional) Merge results to main for persistence
   git checkout main
   git merge retrieval-structured-pipeline --no-ff
```

### Cross-Worktree Comparison Flow

```
1. All worktrees have committed results to their branches

2. From main, run aggregator
   python bmretr/scripts/aggregate_results.py \
     --methods qa-baseline,emb-baseline,structured-pipeline,ensemble \
     --result-dirs bmretr/result/result2,bmretr/result/emb_result2,bmretr/result/structured_result2,bmretr/result/ensemble_result2 \
     --out bmretr/result/comparison/

3. Review comparison report
   - Per-field best method
   - Overall F1 ranking
   - Zero-extraction rates
   - Over-extraction ratios

4. Decide: promote best method to main, or iterate in worktree
```

### Monkey-Patching Contract

The benchmark harness (`run_result2.py`) must support injecting arbitrary retrieval modules. Current implementation:

```python
# Current (hardcoded)
import retrieinfor.main as qa_main
from retrieinfor.main import process_all_cv, process_all_jd
qa_main.CV_RAW = cv_csv
qa_main.JD_RAW = jd_csv
# ...
cv_df = process_all_cv(cv_csv)
```

**Required generalization:**
```python
# Future (configurable)
import importlib.util
spec = importlib.util.spec_from_file_location("method_main", module_path / "main.py")
method_main = importlib.util.module_from_spec(spec)
spec.loader.exec_module(method_main)

method_main.CV_INPUT = cv_csv  # or CV_RAW depending on module
method_main.JD_INPUT = jd_csv
cv_df = method_main.process_all_cv(cv_csv)
```

**Schema contract verification (in bench.py or a new validator):**
```python
EXPECTED_CV_COLS = ["UserID", "summary", "education_level", "institution", "major",
                    "skills", "location", "salary_expectation", "desired_job"]
assert list(pred_cv.columns) == EXPECTED_CV_COLS, f"Schema mismatch: {list(pred_cv.columns)}"
```

---

## Adding a New Retrieval Method

### Step-by-Step Guide

**1. Brainstorm and Design (if novel approach)**
```bash
# Use brainstorming skill to explore approach
# Document in docs/superpowers/specs/YYYY-MM-DD-<method>-design.md
# Get user approval before proceeding
```

**2. Create Worktree**
```bash
# Using superpowers:using-git-worktrees skill
# Or manually:
git worktree add .worktrees/retrieval-my-new-method -b retrieval-my-new-method
cd .worktrees/retrieval-my-new-method
```

**3. Write Worktree README**
```
# .worktrees/retrieval-my-new-method/README.md
# Method: My New Method
# Approach: Brief description
# Target fields: All (or specific subset)
# Expected improvements: List hypotheses
# Status: In progress / Complete / Abandoned
```

**4. Create Method Module**
```
mkdir retrieinfor_mynewmethod
touch retrieinfor_mynewmethod/__init__.py
touch retrieinfor_mynewmethod/main.py
mkdir retrieinfor_mynewmethod/cv retrieinfor_mynewmethod/jd
# Implement per-field extractors
```

**5. Implement with TDD**
- For each field extractor:
  - Write test with synthetic input
  - Verify test fails
  - Implement minimal logic
  - Verify test passes
  - Commit
- Verify full pipeline produces correct schema
- Test on 1-2 real records manually

**6. Integrate with Benchmark**
- Run `run_result2.py` with method path
- Run `bench.py` to compute metrics
- Review per-field results

**7. Commit Results**
```bash
git add retrieinfor_mynewmethod/ bmretr/result/mynewmethod_result2/
git commit -m "feat: my-new-method retrieval + benchmark results"
```

**8. Compare and Decide**
- Run aggregator across all methods
- If this method wins on key fields: consider promoting
- If not: document learnings, abandon or iterate

**9. Cleanup (if abandoned)**
```bash
cd project-root
git worktree remove .worktrees/retrieval-my-new-method
git branch -D retrieval-my-new-method
```

---

## Verification Strategy

### Per-Worktree Verification

**Before committing results from a worktree:**
1. **Schema contract test:** Run a validator that confirms output CSVs have exact required columns
2. **Smoke test:** Run extraction on 1 known record; manually verify output is non-empty for fields that have gold content
3. **Benchmark execution:** `bench.py` completes without errors
4. **Result committed:** `git status` shows no uncommitted changes to method code or result files

### Cross-Method Verification

**When comparing methods:**
1. **Same gold standard:** All methods benchmarked against `gold_mocked/` (or `gold/` for strict comparison)
2. **Same input records:** All methods tested on identical 10-record subset
3. **Identical metric:** Token-overlap F1 (no custom metrics per method)
4. **Reproducible:** Running benchmark twice on same worktree produces identical results

### Superpowers Workflow Verification

**For each task in a retrieval method plan:**
- Implementer subagent reports: tests written, tests passing, self-review complete, committed
- Task reviewer verifies:
  - Spec compliance: "Does this implement the planned extraction logic?"
  - Code quality: "Is code minimal, tested, following project style?"
  - Schema contract: "Does output have required columns?"
- Final branch review (whole worktree):
  - All planned tasks complete
  - Results committed and comparable
  - No modifications to shared `bmretr/` infrastructure

### Regression Prevention

**When modifying benchmark harness (`run_result2.py`, `bench.py`):**
- Re-run all existing methods (qa-baseline, emb-baseline) and verify metrics unchanged
- Add regression test: "Running benchmark on baseline produces F1 within 0.01 of recorded value"

---

## Comparison Mechanism

### Metrics (Existing, Unchanged)

From `bench.py`:
- Per record, per field:
  - `precision = |pred_tokens ∩ gold_tokens| / |pred_tokens|`
  - `recall = |pred_tokens ∩ gold_tokens| / |gold_tokens|`
  - `f1 = 2 * prec * rec / (prec + rec)`
- Macro averages: mean over all records and fields

### Aggregation (New)

**`bmretr/scripts/aggregate_results.py`** (to be implemented) produces:

1. **Cross-method detail:** `comparison/detail_all_methods.csv`
   - Same schema as `detail.csv` but with extra column `method`
   - One row per (method, record, field)

2. **Cross-method summary:** `comparison/summary_all_methods.csv`
   ```
   method,type,mean_precision,mean_recall,mean_f1,n
   qa-baseline,CV,0.XXX,0.XXX,0.XXX,80
   emb-baseline,CV,0.XXX,0.XXX,0.XXX,80
   structured-pipeline,CV,0.XXX,0.XXX,0.XXX,80
   qa-baseline,JD,0.XXX,0.XXX,0.XXX,80
   ...
   ```

3. **Per-field winner:** `comparison/best_per_field.csv`
   ```
   field,best_method,precision,recall,f1,precision_diff_vs_second
   CV.summary,structured-pipeline,0.XXX,0.XXX,0.XXX,0.XXX
   CV.education_level,qa-baseline,0.XXX,0.XXX,0.XXX,0.XXX
   ...
   ```

4. **Human-readable report:** `comparison/report.md`
   - Overall ranking
   - Fields where each method excels
   - Zero-extraction counts per method
   - Over-extraction ratio (pred_tokens/gold_tokens) per method

### Decision Criteria

**Promote a method from worktree to main if:**
- It improves F1 on at least 3 fields by >0.05 without regressing others by >0.02
- OR it reduces zero-extraction rate by >20% on fields with historically high zeros
- AND it passes schema contract and smoke tests

**Keep in worktree for iteration if:**
- Promising on some fields, weak on others
- Shows clear improvement direction but not yet production-ready

**Abandon if:**
- Consistently worse than baselines across all fields
- No clear path to improvement (documented in worktree README)

---

## Directory/Worktree Layout Diagram

```
project-root/
│
├── .gitignore                          # ADD: .worktrees/
├── .worktrees/                         # GIT-IGNORED (verified on creation)
│   │
│   ├── retrieval-qa-baseline/
│   │   ├── retrieinfor/                # Reference copy of QA (or symlink)
│   │   ├── README.md                   # "Baseline QA method. Do not modify."
│   │   └── (no bmretr/ here)
│   │
│   ├── retrieval-emb-baseline/
│   │   ├── retrieinfor1/               # Reference copy of EMB
│   │   ├── README.md
│   │   └── (no bmretr/ here)
│   │
│   ├── retrieval-structured-pipeline/
│   │   ├── retrieinfor_structured/     # NEW module (not overwriting existing)
│   │   │   ├── main.py
│   │   │   ├── section_parser.py
│   │   │   ├── router.py
│   │   │   ├── validators.py
│   │   │   ├── cv/
│   │   │   │   ├── education.py
│   │   │   │   └── ...
│   │   │   └── jd/
│   │   │       └── ...
│   │   ├── README.md                   # Documents approach, hypotheses, status
│   │   └── (no bmretr/ here)
│   │
│   └── retrieval-ensemble/
│       ├── retrieinfor_ensemble/
│       ├── README.md
│       └── (no bmretr/ here)
│
├── bmretr/                             # SHARED — source of truth for benchmarks
│   ├── gold/                           # Human-curated perfect extraction (10 CV + 10 JD)
│   ├── gold_mocked/                    # Gold with empty fields filled
│   ├── raw/                            # Raw input CSVs (cv_first10.csv, jd_first10.csv)
│   ├── inputs/                         # Markdown inputs for run_result2.py (cv.md, jd.md)
│   ├── scripts/
│   │   ├── run_result2.py              # EXTEND: accept --method, --module-path
│   │   ├── bench.py                    # UNCHANGED: generic token-overlap metrics
│   │   ├── create_gold.py              # One-time gold generation
│   │   ├── create_mocked_gold.py       # One-time mocked gold generation
│   │   └── aggregate_results.py        # NEW: cross-method comparison
│   │
│   ├── outputs/                        # Generated by run_result2.py
│   │   ├── qa/                         # From retrieinfor (baseline)
│   │   ├── emb/                        # From retrieinfor1 (baseline)
│   │   ├── structured_result2/         # From worktree experiments
│   │   └── ...
│   │
│   └── result/                         # Generated by bench.py + aggregator
│       ├── result1/
│       ├── result2/                    # QA + EMB baseline results
│       ├── structured_result2/         # Structured pipeline results
│       ├── ensemble_result2/
│       └── comparison/                 # NEW: cross-method summaries
│           ├── detail_all_methods.csv
│           ├── summary_all_methods.csv
│           ├── best_per_field.csv
│           └── report.md
│
├── retrieinfor/                        # On main branch (baseline reference)
│   ├── main.py
│   ├── qa_model.py
│   ├── cv/
│   └── jd/
│
├── retrieinfor1/                       # On main branch (baseline reference)
│   ├── main.py
│   ├── embedding_model.py
│   ├── cv/
│   └── jd/
│
└── docs/
    └── superpowers/
        ├── specs/                      # Design docs (from brainstorming)
        │   └── YYYY-MM-DD-<method>-design.md
        └── plans/                      # Implementation plans (from writing-plans)
            └── 2026-07-02-git-worktree-architecture-for-retrieval-method-experiments.md
```

---

## Integration Points with Superpowers Skills

### 1. Using Git Worktrees

**When to invoke:** Before ANY work on a new retrieval method.

**Invocation:**
```
I'm using the using-git-worktrees skill to set up an isolated workspace.
```

**What it does:**
- Detects if already in isolated workspace (Step 0)
- Creates `.worktrees/` if needed (Step 1b)
- Verifies `.worktrees/` is git-ignored
- Auto-detects Python project, suggests `pip install`
- Verifies clean baseline (tests pass)

**Output:**
```
Worktree ready at /path/to/project/.worktrees/retrieval-structured-pipeline
Tests passing (N tests, 0 failures)
Ready to implement <method-name>
```

### 2. Brainstorming

**When to invoke:** For novel retrieval approaches (not just re-running baselines).

**Flow:**
1. Explore project context (read this architecture doc, PATTERN_ANALYSIS_AND_SOLUTIONS.md, existing code)
2. Ask clarifying questions (one at a time)
3. Propose 2-3 approaches with trade-offs
4. Present design sections, get approval per section
5. Write design doc to `docs/superpowers/specs/YYYY-MM-DD-<method>-design.md`
6. Spec self-review
7. User reviews spec
8. Transition to writing-plans

**Hard gate:** Do NOT write code or create worktree until design is approved.

### 3. Writing Plans

**When to invoke:** After brainstorming approval OR for straightforward extensions of existing patterns.

**Save to:** `docs/superpowers/plans/YYYY-MM-DD-<method>-implementation.md`

**Plan must:**
- Reference this architecture document
- Specify worktree name and module path
- Decompose into TDD tasks (failing test → minimal code → pass → commit)
- Include schema contract verification steps
- Include benchmark integration steps
- Include result commit step

### 4. Subagent-Driven Development (or Executing Plans)

**When to invoke:** To implement the plan created by writing-plans.

**Subagent-driven (preferred):**
- Fresh subagent per task
- Task reviewer after each task (spec + quality + schema contract)
- Final whole-branch review
- Use `superpowers:test-driven-development` within each subagent

**Executing plans (alternative):**
- For simpler methods
- Still requires worktree isolation
- Batch execution with checkpoints

### 5. Finishing a Development Branch

**When to invoke:** After all tasks complete and verified.

**Actions:**
- Verify tests pass
- Verify results committed
- Present options: merge to main, keep in worktree, abandon
- Execute chosen action

---

## Development Lifecycle

### Phase 1: Setup (One-Time)

- [ ] Add `.worktrees/` to `.gitignore` (if not already)
- [ ] Verify `.worktrees/` is ignored: `git check-ignore -q .worktrees`
- [ ] Extend `run_result2.py` to accept `--method`, `--module-path`, `--output-dir`
- [ ] Add schema contract validator (function or script)
- [ ] Create `bmretr/scripts/aggregate_results.py` skeleton
- [ ] Document baseline results in `bmretr/result/result2/` (already done)

### Phase 2: Per-Method Experiment (Repeatable)

For each retrieval method:

1. **Brainstorm (if novel)**
   - [ ] Use `superpowers:brainstorming`
   - [ ] Write design doc
   - [ ] Get user approval

2. **Create Worktree**
   - [ ] Use `superpowers:using-git-worktrees`
   - [ ] Name: `retrieval-{method}`
   - [ ] Write `README.md` documenting approach

3. **Plan Implementation**
   - [ ] Use `superpowers:writing-plans`
   - [ ] Decompose into TDD tasks
   - [ ] Reference schema contract

4. **Implement**
   - [ ] Use `superpowers:subagent-driven-development` or `executing-plans`
   - [ ] Follow TDD per task
   - [ ] Verify schema contract at integration points

5. **Benchmark**
   - [ ] Run `run_result2.py` with method path
   - [ ] Run `bench.py`
   - [ ] Review per-field results

6. **Commit Results**
   - [ ] Commit method code + test results
   - [ ] Commit benchmark outputs to worktree branch

7. **Compare (optional, after multiple methods)**
   - [ ] Run `aggregate_results.py`
   - [ ] Review cross-method report

8. **Decide**
   - [ ] Promote to main (if wins)
   - [ ] Iterate in worktree (if promising)
   - [ ] Abandon (if consistently worse)

### Phase 3: Promotion (When a Method Wins)

- [ ] Merge worktree branch to main: `git merge retrieval-{method} --no-ff`
- [ ] Update main branch `retrieinfor/` or `retrieinfor1/` (or add new module)
- [ ] Update `run_result2.py` to include new method in default runs
- [ ] Update documentation (README, CLAUDE.md if relevant)
- [ ] Remove worktree (optional): `git worktree remove .worktrees/retrieval-{method}`

---

## Risks and Mitigations

### Risk 1: Worktree contents accidentally committed to main

**Mitigation:**
- `.gitignore` entry for `.worktrees/`
- `git check-ignore` verification in using-git-worktrees skill
- Worktree README explicitly states: "Do not commit bmretr/ from here"

### Risk 2: Benchmark harness modified from within worktree

**Mitigation:**
- Architecture doc states: "Do NOT modify bmretr/ from within worktrees"
- Superpowers task reviewer checks for `bmretr/` diffs in worktree changes
- Final branch review includes: "No modifications to shared bmretr/"

### Risk 3: Schema drift (different methods produce different columns)

**Mitigation:**
- Explicit schema contract in this document
- Schema validator run before benchmark
- Task reviewer checks: "Output CSV has exact required columns"

### Risk 4: Results not reproducible across worktrees

**Mitigation:**
- All methods use same gold, same inputs, same metric
- Reproducibility verification: run baseline twice, confirm identical F1
- Document random seeds if any (currently none — models are deterministic given same input)

### Risk 5: Worktree proliferation (too many abandoned experiments)

**Mitigation:**
- Worktree README must include "Status" field
- Quarterly cleanup: list all worktrees, prompt human to keep/abandon
- Git branch naming makes it easy to identify: `git branch -a | grep retrieval-`

---

## Success Criteria

**This architecture is successful if:**

1. An engineer unfamiliar with the codebase can:
   - Read this document
   - Create a worktree for a new retrieval method
   - Implement the method following TDD
   - Run benchmarks and get comparable results
   - All without modifying shared `bmretr/` infrastructure

2. Multiple retrieval methods can be developed in parallel without conflicts

3. Results are reproducible: same method + same inputs → same metrics

4. Comparison across methods is automated and actionable (per-field winners, overall ranking)

5. Superpowers skills are integrated at each lifecycle stage (worktree creation, planning, implementation, review, completion)

6. Baseline performance is preserved: running `run_result2.py` + `bench.py` on existing `retrieinfor/` and `retrieinfor1/` produces unchanged metrics

---

## Open Questions (for Future Resolution)

1. **Should worktrees use symlinks to share `retrieinfor/` baseline code, or copy it?**
   - Current recommendation: copy for isolation, document that baseline should not be modified
   - Alternative: symlink for DRY, but risk of accidental modification

2. **Should `run_result2.py` be generalized to a config-driven runner?**
   - Current: extend with CLI args
   - Future: `bmretr/config/methods.yaml` listing active methods and paths

3. **How to handle methods that only target a subset of fields?**
   - Example: "improved education extraction only"
   - Options:
     - Method must still produce all columns (fill others with baseline or empty)
     - Benchmark harness supports partial results (complexity)
   - Recommendation: require full schema; partial improvements can be composed via ensemble later

4. **Should results be stored in worktree or copied to main's `bmretr/result/`?**
   - Current recommendation: commit to worktree, optionally merge to main
   - Alternative: always copy to main's result dir with method prefix
   - Trade-off: worktree keeps results with code; main keeps all results in one place

---

## References

- `bmretr/README.md` — Benchmark infrastructure documentation
- `bmretr/PATTERN_ANALYSIS_AND_SOLUTIONS.md` — Problem analysis and proposed solutions
- `bmretr/result/result2/PATTERN_ANALYSIS_AND_SOLUTIONS.md` — (duplicate location, same content)
- `.claude/skills/superpowers/skills/using-git-worktrees/SKILL.md` — Worktree skill
- `.claude/skills/superpowers/skills/writing-plans/SKILL.md` — Plan writing skill
- `.claude/skills/superpowers/skills/subagent-driven-development/SKILL.md` — Subagent execution
- `.claude/skills/superpowers/skills/executing-plans/SKILL.md` — Inline plan execution
- `.claude/skills/superpowers/skills/brainstorming/SKILL.md` — Design exploration
- `retrieinfor/main.py` — QA orchestrator
- `retrieinfor1/main.py` — EMB orchestrator
- `bmretr/scripts/run_result2.py` — Current benchmark runner
- `bmretr/scripts/bench.py` — Current metrics computation

---

## Appendix: Example Task Breakdown (for a Hypothetical Structured Pipeline Method)

This illustrates how a writing-plans output would look for a concrete method. This is NOT part of the architecture to implement — it is an example of what a plan for a specific method would contain.

### Task 1: Section Parser

**Files:**
- Create: `retrieinfor_structured/section_parser.py`
- Test: `retrieinfor_structured/test_section_parser.py`

**Interfaces:**
- Produces: `parse_sections(text: str) -> List[Section]` where `Section = {header: str, content: str, start_line: int, end_line: int}`

- [ ] **Step 1:** Write failing test for `parse_sections` on sample CV text with headers "Học vấn:", "Kỹ năng:"
- [ ] **Step 2:** Run test, verify FAIL
- [ ] **Step 3:** Implement regex-based section splitter
- [ ] **Step 4:** Run test, verify PASS
- [ ] **Step 5:** Commit

### Task 2: Router

**Files:**
- Create: `retrieinfor_structured/router.py`
- Test: `retrieinfor_structured/test_router.py`

**Interfaces:**
- Consumes: `Section` list from Task 1
- Produces: `route_field(sections, field_name) -> Section` or raises if no match

- [ ] ... (TDD steps)

### Task 3: Education Validator

**Files:**
- Modify: `retrieinfor_structured/validators.py`
- Test: `retrieinfor_structured/test_validators.py`

**Interfaces:**
- Produces: `validate_education_level(text: str) -> str` (normalized or "")

- [ ] ... (TDD steps)

### Task 4: Main Orchestrator

**Files:**
- Create: `retrieinfor_structured/main.py`
- Test: `retrieinfor_structured/test_main.py`

**Interfaces:**
- Produces: `process_all_cv(input_csv) -> pd.DataFrame` with exact CV schema columns
- Produces: `process_all_jd(input_csv) -> pd.DataFrame` with exact JD schema columns

- [ ] ... (TDD steps, including schema contract test)

### Task 5: Benchmark Integration

**Files:**
- No new files (uses existing bmretr/scripts)

- [ ] **Step 1:** Run `python bmretr/scripts/run_result2.py --method structured --module-path .worktrees/retrieval-structured-pipeline/retrieinfor_structured --output-dir bmretr/outputs/structured_result2`
- [ ] **Step 2:** Verify outputs exist and have correct schema
- [ ] **Step 3:** Run `python bmretr/scripts/bench.py` with appropriate paths
- [ ] **Step 4:** Verify metrics computed, review per-field results
- [ ] **Step 5:** Commit results to worktree branch

---

**End of Architecture Plan**

This document is the input to `superpowers:writing-plans` when creating implementation plans for specific retrieval methods or for the benchmark harness extensions described herein.