#!/usr/bin/env python3
r"""
Statistics behind Figure 4a and the sensitivity results (no plotting).

Input: the tidy table written by summary_statistics.py (all_datasets_tidy_<judge>.csv).
Writes:
  <dataset>_SensitivityDistribution_<Edit>_stacked_p012_counts.csv
      counts and proportions of sensitivity scores 0/1/2 per summarizer x prompt, per dataset
  AllDatasets_SensitivityDistribution_<Edit>_stacked_p012_counts.csv
      the same pooled over datasets (the numbers reported in the Results)
  AllDatasets_SensitivityDistribution_<Edit>_long_vs_short_perm_tests.csv
      Detailed vs Concise permutation test of the whole 0/1/2 distribution
      (chi-square statistic on the 2x3 table, 20,000 permutations)

All statistical functions are copied verbatim from the scripts used for the paper;
only the driver at the bottom is new.

Usage:
  python analysis/sensitivity_statistics.py --tidy results/gpt_5/all_datasets_tidy_gpt_5.csv \
         --out-dir results/gpt_5
"""
import argparse
import os
from typing import List

import numpy as np
import pandas as pd


# ======================================================================================
# Statistics
# (verbatim from llm_as_judge_prompt_effects.py)
# ======================================================================================
PROMPT_ORDER = ["Long", "Short", "Unknown"]

def ensure_order(cat: pd.Series, order: List[str]) -> pd.Series:
    present = [x for x in order if x in set(cat)]
    return pd.Categorical(cat, categories=present, ordered=True)

def _chi2_stat_2xk(counts_a: np.ndarray, counts_b: np.ndarray) -> float:
    """
    Pearson chi-square statistic for a 2×K table.
    counts_a, counts_b are length-K arrays (e.g., K=3 for 0/1/2).
    """
    counts_a = np.asarray(counts_a, dtype=float)
    counts_b = np.asarray(counts_b, dtype=float)
    obs = np.vstack([counts_a, counts_b])  # shape (2, K)

    row_sums = obs.sum(axis=1, keepdims=True)
    col_sums = obs.sum(axis=0, keepdims=True)
    total = obs.sum()
    if total <= 0:
        return np.nan

    exp = (row_sums @ col_sums) / total
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum((obs - exp) ** 2 / exp)
    return float(chi2)

def _perm_test_sensitivity_distribution(
    scores: np.ndarray,         # values in {0,1,2}
    prompts: np.ndarray,        # strings "Long"/"Short"
    B: int = 20000,
    seed: int = 123,
) -> float:
    """
    Two-sided permutation test for whether the 0/1/2 distribution differs between Long and Short.
    Test statistic: chi-square on 2×3 table.
    """
    scores = np.asarray(scores)
    prompts = np.asarray(prompts)

    mask = np.isin(np.char.lower(prompts.astype(str)), ["long", "short"])
    scores = scores[mask]
    prompts = prompts[mask]

    if scores.size == 0:
        return np.nan

    is_long = np.char.lower(prompts.astype(str)) == "long"
    is_short = np.char.lower(prompts.astype(str)) == "short"
    if is_long.sum() == 0 or is_short.sum() == 0:
        return np.nan

    # observed
    counts_long = np.array([(scores[is_long] == v).sum() for v in (0, 1, 2)], dtype=int)
    counts_short = np.array([(scores[is_short] == v).sum() for v in (0, 1, 2)], dtype=int)
    obs_stat = _chi2_stat_2xk(counts_long, counts_short)
    if np.isnan(obs_stat):
        return np.nan

    # permute labels
    rng = np.random.default_rng(seed)
    long_n = int(is_long.sum())
    pooled_scores = scores.copy()

    cnt = 0
    for _ in range(B):
        perm = rng.permutation(scores.size)
        long_idx = perm[:long_n]
        short_idx = perm[long_n:]

        cL = np.array([(pooled_scores[long_idx] == v).sum() for v in (0, 1, 2)], dtype=int)
        cS = np.array([(pooled_scores[short_idx] == v).sum() for v in (0, 1, 2)], dtype=int)
        st = _chi2_stat_2xk(cL, cS)

        if st >= obs_stat:
            cnt += 1

    return (cnt + 1.0) / (B + 1.0)

# ======================================================================================
# Output helpers
# (verbatim from llm_as_judge_plotting.py)
# ======================================================================================
def _csv_path_next_to_plot(plot_path: str, suffix: str) -> str:
    base, _ = os.path.splitext(plot_path)
    return f"{base}{suffix}.csv"

def _save_csv_next_to_plot(df: pd.DataFrame, plot_path: str, suffix: str) -> str:
    csv_path = _csv_path_next_to_plot(plot_path, suffix)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    return csv_path

def _p_to_stars(p: float) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "p=NA"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p <= 0.05:
        return "*"
    return f"p={p:.2f}"

# ==========================================================================================
# Driver (new; no plotting). Statistical lines are those of plot_sensitivity_distribution
# and plot_sensitivity_distribution_combined.
# ==========================================================================================
def _stacks(sub: pd.DataFrame, global_summarizers: List[str]) -> pd.DataFrame:
    stacks = []
    for s in global_summarizers:
        for p in ["Long", "Short"]:
            g = sub[(sub["summarizer"] == s) & (sub["prompt"].str.lower() == p.lower())]["score"]
            if g.empty:
                stacks.append((s, p, 0, 0, 0, 0, 0, 0, 0)); continue
            counts = g.value_counts().to_dict()
            n  = int(len(g))
            c0 = int(counts.get(0.0, 0))
            c1 = int(counts.get(1.0, 0))
            c2 = int(counts.get(2.0, 0))
            stacks.append((s, p, c0 / n, c1 / n, c2 / n, n, c0, c1, c2))
    sd = pd.DataFrame(stacks, columns=["summarizer", "prompt", "p0", "p1", "p2", "n", "c0", "c1", "c2"])
    sd["prompt"] = ensure_order(sd["prompt"], PROMPT_ORDER)
    return sd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tidy", required=True, help="all_datasets_tidy_<judge>.csv from summary_statistics.py")
    ap.add_argument("--out-dir", required=True, help="Output folder for the tables")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    tidy = pd.read_csv(args.tidy)
    global_summarizers = sorted(tidy["summarizer"].dropna().unique().tolist())
    df = tidy[tidy["metric"] == "Sensitivity to Changes"].copy()
    edit_types = sorted(df["edit_type"].dropna().unique())

    # per dataset
    for ds in sorted(df["dataset"].dropna().unique()):
        dsd = df[df["dataset"] == ds]
        for ed in edit_types:
            sub = dsd[dsd["edit_type"] == ed]
            if sub.empty:
                continue
            _export = _stacks(sub, global_summarizers)
            _export.insert(0, "edit_type", ed)
            _export.insert(0, "dataset", ds)
            _save_csv_next_to_plot(_export, os.path.join(args.out_dir, f"{ds}_SensitivityDistribution_{ed.replace(' ','')}"),
                                   "_stacked_p012_counts")

    # pooled across datasets + Long vs Short permutation tests
    for ed in edit_types:
        sub = df[df["edit_type"] == ed]
        if sub.empty:
            continue
        sd = _stacks(sub, global_summarizers)
        full_by_summarizer = sd.groupby("summarizer")["p2"].mean().reindex(global_summarizers)
        ordered_summarizers = full_by_summarizer.sort_values(ascending=True).index.tolist()
        _p_rows = []
        for s in ordered_summarizers:
            ss = sub[(sub["summarizer"] == s) & (sub["prompt"].str.lower().isin(["long", "short"]))].copy()
            if ss.empty:
                continue
            pval = _perm_test_sensitivity_distribution(
                scores=ss["score"].to_numpy(), prompts=ss["prompt"].to_numpy(), B=20000, seed=123)
            cL = [(ss[(ss["prompt"].str.lower() == "long")]["score"] == v).sum() for v in (0, 1, 2)]
            cS = [(ss[(ss["prompt"].str.lower() == "short")]["score"] == v).sum() for v in (0, 1, 2)]
            _p_rows.append({
                "dataset": "AllDatasets", "edit_type": ed, "summarizer": s,
                "p_value_perm_chi2_2x3": pval, "label": _p_to_stars(float(pval)),
                "long_c0": int(cL[0]), "long_c1": int(cL[1]), "long_c2": int(cL[2]),
                "short_c0": int(cS[0]), "short_c1": int(cS[1]), "short_c2": int(cS[2]),
                "B": 20000,
            })
        base = os.path.join(args.out_dir, f"AllDatasets_SensitivityDistribution_{ed.replace(' ','')}")
        _export = sd.copy()
        _export.insert(0, "edit_type", ed)
        _export.insert(0, "dataset", "AllDatasets")
        _save_csv_next_to_plot(_export, base, "_stacked_p012_counts")
        if _p_rows:
            _save_csv_next_to_plot(pd.DataFrame(_p_rows), base, "_long_vs_short_perm_tests")
    print(f"[OK] tables written to {args.out_dir}")


if __name__ == "__main__":
    main()
