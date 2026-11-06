#!/usr/bin/env python3
r"""
Build the report-level long dataset for the multi-factor models (fit_models.py).

Reads the final GPT-5 judge score files
    <scores-dir>/gpt_5_<dataset>_<long|short>_<alter_terms|missing_facts>_combined_cleaned_wide.csv
and writes one long-format table (72,000 rows) with a report identifier, which the
tidy table from analysis/summary_statistics.py does not carry but a report-level random
effect / cluster-robust variance estimator needs.

Design (verified in verify_design()):
    300 unique reports (100 per dataset), the SAME report ids appearing in every
    prompt x edit-type file, fully crossed with
        dataset(3) x prompt(2) x summarizer(6) x condition(5)
    where condition = edit_type x difficulty:
        AlterTerm-Easy / -Medium / -Hard, MissingFact-Easy / -Hard
    -> 3 * 2 * 6 * 5 * 100 = 18,000 observations per metric, 60 per report.

Outcome coding (IMPORTANT):
    factual_faithfulness    1 = fact preserved            (higher is better)
    certainty_fidelity      1 = certainty maintained      (higher is better)
    sensitivity_to_changes  0/1/2 ordinal                 (higher is better)
    hallucination           1 = summary avoided a new finding, i.e. NO hallucination.
        The manuscript reports a *hallucination rate*, so this script additionally
        emits `outcome` = 1 - score for this metric, named 'Hallucination (present)'.
        Optionally cross-checked against the hallucination rates written by
        analysis/summary_statistics.py (--reference-hallucination-tests).

Usage:
    python analysis/multifactor_models/build_model_dataset.py \
        --scores-dir data/judge_scores/gpt_5 --out-dir results/multifactor_models \
        [--reference-tidy results/gpt_5/all_datasets_tidy_gpt_5.csv] \
        [--reference-hallucination-tests results/gpt_5/HallucinationRates_AllDatasets_gpt_5_long_vs_short_perm_tests.csv]
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

# Set from the command line in main().
WIDE_DIR = "data/judge_scores/gpt_5"
REFERENCE_TIDY = None   # optional tidy table from analysis/summary_statistics.py
HALL_TESTS = None       # optional hallucination test table from analysis/summary_statistics.py
OUT_DIR = "results/multifactor_models"
OUT_PATH = os.path.join(OUT_DIR, "model_dataset_gpt5_judge.csv")

JUDGE = "gpt_5"

# Only score files produced by the GPT-5 judge are read.
FILE_PREFIX = "gpt_5_"

METRIC_RAW_TO_NICE = {
    "hallucination": "Hallucination",
    "sensitivity_to_changes": "Sensitivity to Changes",
    "factual_faithfulness": "Factual Faithfulness",
    "certainty_fidelity": "Certainty Fidelity",
}

SUMMARIZER_TOKENS = ["llama3_1", "llama3_3", "llama_3_1", "llama_3_3", "gpt", "qwen", "claude", "medgemma"]
JUDGE_TOKENS = ["llama_3_1", "llama3_1", "gpt", "gpt_5", "medgemma"]
SUMMARIZER_OPT_SUFFIX = r"(?:_llama3|_instruct|_chat|_it)?"

# Same column grammar as llm_as_judge_plotting.py (copied, not imported: that module
# executes its full plotting pipeline at import time).
COL_RE = re.compile(
    r"^(?P<metric>hallucination|sensitivity_to_changes|factual_faithfulness|certainty_fidelity)_"
    r"(?P<edit>missing|missing_facts|alter|alter_terms)_"
    r"(?P<difficulty>easy|medium|hard)_"
    r"(?P<summarizer>(" + "|".join(map(re.escape, SUMMARIZER_TOKENS)) + r"))"
    + SUMMARIZER_OPT_SUFFIX +
    r"_(?P<judge>(" + "|".join(map(re.escape, JUDGE_TOKENS)) + r"))$",
    re.IGNORECASE,
)

# Display names used in the manuscript.
SUMMARIZER_LABEL = {
    "claude": "Claude Sonnet 4.5",
    "gpt": "GPT-5-mini",
    "llama_3_1": "Llama 3.1 (8B)",
    "llama_3_3": "Llama 3.3 (70B)",
    "medgemma": "MedGemma 27B",
    "qwen": "Qwen 3 (4B)",
}
PROMPT_LABEL = {"Long": "Detailed", "Short": "Concise"}
DATASET_LABEL = {"inspect": "InSpect", "merlin": "MERLIN", "mimic": "MIMIC-IV"}

CONDITION_ORDER = [
    "AlterTerm-Easy",
    "AlterTerm-Medium",
    "AlterTerm-Hard",
    "MissingFact-Easy",
    "MissingFact-Hard",
]


def standardize_summarizer(s: str) -> str:
    s = s.lower()
    s = re.sub(r"_(llama3|instruct|chat|it)$", "", s)
    s = s.replace("llama3_3", "llama_3_3").replace("llama3_1", "llama_3_1")
    return s


def standardize_edit(x: str) -> str:
    x = x.lower()
    if x in ("missing_facts", "missing"):
        return "missing"
    if x in ("alter_terms", "alter"):
        return "alter"
    return x


def parse_meta_from_filename(fp: str):
    base = os.path.basename(fp).lower()
    dataset = "unknown"
    for d in ("inspect", "merlin", "mimic"):
        if f"{d}_" in base:
            dataset = d
            break
    prompt = "Long" if "long" in base else "Short" if "short" in base else "Unknown"
    return dataset, prompt


def report_id_column(df: pd.DataFrame) -> pd.Series:
    """MERLIN/InSpect use 'study id'; MIMIC-IV uses 'study_id'."""
    for c in ("study id", "study_id"):
        if c in df.columns:
            return df[c].astype(str)
    raise KeyError(f"No report identifier column found; have {list(df.columns)[:5]}")


def build_long(files) -> pd.DataFrame:
    parts = []
    for fp in files:
        dataset, prompt = parse_meta_from_filename(fp)
        df = pd.read_csv(fp)
        rid = dataset + ":" + report_id_column(df)
        for col in df.columns:
            m = COL_RE.match(col)
            if not m:
                continue
            edit = standardize_edit(m.group("edit"))
            diff = m.group("difficulty").lower()
            # Impression removal was only generated at easy/hard; the plotting
            # pipeline drops missing-facts 'medium' and so do we.
            if edit == "missing" and diff == "medium":
                continue
            vals = pd.to_numeric(df[col], errors="coerce")
            keep = vals.notna()
            if not keep.any():
                continue
            parts.append(
                pd.DataFrame(
                    {
                        "report_id": rid[keep].values,
                        "dataset": dataset,
                        "prompt": prompt,
                        "edit_type": "Alter Term" if edit == "alter" else "Missing Fact",
                        "difficulty": diff,
                        "summarizer": standardize_summarizer(m.group("summarizer")),
                        "metric": METRIC_RAW_TO_NICE[m.group("metric").lower()],
                        "score": vals[keep].astype(float).values,
                    }
                )
            )
    long = pd.concat(parts, ignore_index=True)

    long["condition"] = (
        long["edit_type"].map({"Alter Term": "AlterTerm", "Missing Fact": "MissingFact"})
        + "-"
        + long["difficulty"].str.capitalize()
    )
    # Analysis outcome: flip hallucination so 1 = hallucination PRESENT, matching the
    # direction reported in the manuscript.
    long["outcome"] = np.where(long["metric"] == "Hallucination", 1.0 - long["score"], long["score"])
    long["outcome_name"] = np.where(
        long["metric"] == "Hallucination", "Hallucination (present)", long["metric"]
    )

    long["summarizer_label"] = long["summarizer"].map(SUMMARIZER_LABEL)
    long["prompt_label"] = long["prompt"].map(PROMPT_LABEL)
    long["dataset_label"] = long["dataset"].map(DATASET_LABEL)
    long["judge"] = JUDGE

    cols = [
        "judge", "report_id", "dataset", "dataset_label", "prompt", "prompt_label",
        "summarizer", "summarizer_label", "edit_type", "difficulty", "condition",
        "metric", "outcome_name", "score", "outcome",
    ]
    return long[cols]


def verify_design(long: pd.DataFrame) -> list[str]:
    """Hard checks on the factorial structure. Raises if the design is not as assumed."""
    msgs = []
    n_reports = long["report_id"].nunique()
    assert n_reports == 300, f"expected 300 unique reports, got {n_reports}"
    msgs.append(f"unique reports: {n_reports} (100 per dataset)")

    per_ds = long.groupby("dataset")["report_id"].nunique().to_dict()
    assert set(per_ds.values()) == {100}, per_ds
    msgs.append(f"reports per dataset: {per_ds}")

    for met, g in long.groupby("metric"):
        assert len(g) == 18000, f"{met}: expected 18000 rows, got {len(g)}"
        cell = g.groupby(["dataset", "prompt", "summarizer", "condition"]).size()
        assert cell.nunique() == 1 and cell.iloc[0] == 100, (
            f"{met}: design not balanced; cell sizes {sorted(cell.unique())}"
        )
        assert len(cell) == 3 * 2 * 6 * 5, f"{met}: expected 180 cells, got {len(cell)}"
        per_report = g.groupby("report_id").size()
        assert per_report.nunique() == 1 and per_report.iloc[0] == 60
    msgs.append("per metric: 18,000 rows = 180 balanced cells x 100 reports; 60 obs per report")

    assert long["outcome"].notna().all(), "missing outcomes present"
    for met, g in long.groupby("metric"):
        vals = sorted(g["score"].unique())
        expected = [0.0, 1.0, 2.0] if met == "Sensitivity to Changes" else [0.0, 1.0]
        assert vals == expected, f"{met}: unexpected score values {vals}"
    msgs.append("outcome value ranges as expected (binary; ordinal 0/1/2 for sensitivity)")
    return msgs


def verify_against_reference(long: pd.DataFrame) -> list[str]:
    """Reproduce analysis/all_datasets_tidy_gpt_5.csv cell-by-cell (counts and means)."""
    if not REFERENCE_TIDY or not os.path.exists(REFERENCE_TIDY):
        return ["[WARN] reference tidy file not found; skipped equivalence check"]
    ref = pd.read_csv(REFERENCE_TIDY)
    keys = ["dataset", "prompt", "edit_type", "metric", "difficulty", "summarizer"]
    a = ref.groupby(keys)["score"].agg(["size", "sum", "mean"]).sort_index()
    b = long.groupby(keys)["score"].agg(["size", "sum", "mean"]).sort_index()
    assert a.index.equals(b.index), "cell keys differ from the published tidy file"
    assert (a["size"].values == b["size"].values).all(), "cell counts differ"
    assert np.allclose(a["sum"].values, b["sum"].values), "cell score sums differ"
    max_dev = float(np.max(np.abs(a["mean"].values - b["mean"].values)))
    return [
        f"equivalence vs {os.path.basename(REFERENCE_TIDY)}: {len(a)} cells, "
        f"counts and sums identical, max |mean difference| = {max_dev:.2e}"
    ]


def verify_hallucination_polarity(long: pd.DataFrame) -> list[str]:
    """Cross-check the flip against the published permutation-test rates."""
    ref = HALL_TESTS
    if not ref or not os.path.exists(ref):
        return ["[WARN] published hallucination-rate file not found; skipped polarity check"]
    pub = pd.read_csv(ref)
    pub = pub[pub["judge"] == JUDGE]
    h = long[long["metric"] == "Hallucination"]
    ours = (
        h.groupby(["edit_type", "summarizer", "prompt"])["outcome"].mean().mul(100).reset_index()
    )
    pub_long = pub[["edit_type", "summarizer", "rate_long_pct", "rate_short_pct"]].melt(
        id_vars=["edit_type", "summarizer"], var_name="prompt", value_name="pub_pct"
    )
    pub_long["prompt"] = pub_long["prompt"].map({"rate_long_pct": "Long", "rate_short_pct": "Short"})
    mg = ours.merge(pub_long, on=["edit_type", "summarizer", "prompt"], how="inner")
    assert len(mg) == 24, f"expected 24 published rate comparisons, got {len(mg)}"
    dev = float(np.max(np.abs(mg["outcome"] - mg["pub_pct"])))
    assert dev < 1e-9, f"hallucination rates disagree with published values (max dev {dev})"
    return [
        f"hallucination polarity: 1 - score reproduces all {len(mg)} published "
        f"rate_long/rate_short values exactly (max dev {dev:.1e} pp)"
    ]


def main() -> int:
    global WIDE_DIR, REFERENCE_TIDY, HALL_TESTS, OUT_DIR, OUT_PATH
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores-dir", required=True, help="Folder with the gpt_5_*_combined_cleaned_wide.csv score files")
    ap.add_argument("--out-dir", default="results/multifactor_models")
    ap.add_argument("--reference-tidy", default=None)
    ap.add_argument("--reference-hallucination-tests", default=None)
    args = ap.parse_args()
    WIDE_DIR, REFERENCE_TIDY, HALL_TESTS = args.scores_dir, args.reference_tidy, args.reference_hallucination_tests
    OUT_DIR = args.out_dir
    OUT_PATH = os.path.join(OUT_DIR, "model_dataset_gpt5_judge.csv")
    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted(
        os.path.join(WIDE_DIR, f)
        for f in os.listdir(WIDE_DIR)
        if f.startswith(FILE_PREFIX) and f.endswith("_combined_cleaned_wide.csv")
    )
    print(f"[INFO] {len(files)} GPT-5 judge score files:")
    for f in files:
        print("   ", os.path.basename(f))
    if len(files) != 12:
        print(f"[WARN] expected 12 files (3 datasets x 2 prompts x 2 edit types), got {len(files)}")

    long = build_long(files)

    print("\n[VERIFY]")
    for m in verify_design(long) + verify_against_reference(long) + verify_hallucination_polarity(long):
        print("   ", m)

    long.to_csv(OUT_PATH, index=False)
    print(f"\n[OK] wrote {OUT_PATH}  ({len(long):,} rows)")
    print("\n[SUMMARY] mean outcome by metric x summarizer (rows = analysis outcome direction)")
    piv = long.pivot_table(index="outcome_name", columns="summarizer_label", values="outcome", aggfunc="mean")
    print(piv.round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
