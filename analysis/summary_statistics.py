#!/usr/bin/env python3
r"""
Statistics behind Figures 3 and 4b and the Results text (no plotting).

Reads the final judge score files and writes:
  all_datasets_tidy_<judge>.csv
      one row per (dataset, prompt, edit type, metric, difficulty, summarizer, report)
  FactualFaithfulness_AllDatasets_<Edit>_SummarizerPrompt_bar_stats.csv            (Figure 3b)
      mean factual faithfulness per summarizer x prompt with 95% Wilson CI
  FactualFaithfulness_AllDatasets_<Edit>_SummarizerPrompt_long_vs_short_perm_tests.csv  (Figure 3b)
      Detailed (Long) vs Concise (Short) two-sided permutation test (20,000 permutations)
  <dataset|AllDatasets>_<Edit>_Factual_vs_Certainty_scatter_points_stats.csv       (Figure 3a)
      factual faithfulness (x) and certainty fidelity (y) with 95% Wilson CIs
  HallucinationRates_AllDatasets_<judge>_rates_by_prompt.csv                        (Figure 4b)
      hallucination rate (%) per summarizer x prompt with 95% Wilson CI
  HallucinationRates_AllDatasets_<judge>_long_vs_short_perm_tests.csv              (Figure 4b)
      Detailed vs Concise permutation test on the hallucination indicator (20,000 permutations)

Score coding: hallucination score 1 = NO hallucination, so a summary is counted as
hallucinated when score == 0. Prompt labels: Long = detailed prompt, Short = concise prompt.

All statistical functions below are copied verbatim from the scripts used for the paper;
only the driver at the bottom is new.

Usage:
  python analysis/summary_statistics.py --scores-dir data/judge_scores/gpt_5 --judge gpt_5 \
         --file-order analysis/file_order_gpt_5.txt --out-dir results/gpt_5
"""
import argparse
import glob
import math
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ======================================================================================
# Score-file parsing
# (verbatim from llm_as_judge_plotting.py)
# ======================================================================================
METRIC_RAW_TO_NICE = {
    "hallucination": "Hallucination",
    "sensitivity_to_changes": "Sensitivity to Changes",
    "factual_faithfulness": "Factual Faithfulness",
    "certainty_fidelity": "Certainty Fidelity",
}

JUDGE_TOKENS = ["llama_3_1", "llama3_1", "gpt", "gpt_5", "medgemma"]

SUMMARIZER_TOKENS = ["llama3_1", "llama3_3", "llama_3_1", "llama_3_3", "gpt", "qwen", "claude", "medgemma"]

SUMMARIZER_OPT_SUFFIX = r"(?:_llama3|_instruct|_chat|_it)?"

COL_RE = re.compile(
    r"^(?P<metric>hallucination|sensitivity_to_changes|factual_faithfulness|certainty_fidelity)_"
    r"(?P<edit>missing|missing_facts|alter|alter_terms)_"
    r"(?P<difficulty>easy|medium|hard)_"
    r"(?P<summarizer>(" + "|".join(map(re.escape, SUMMARIZER_TOKENS)) + r"))"
    + SUMMARIZER_OPT_SUFFIX +
    r"_(?P<judge>(" + "|".join(map(re.escape, JUDGE_TOKENS)) + r"))$",
    re.IGNORECASE
)

def parse_meta_from_filename(fp: str):
    base = os.path.basename(fp).lower()
    dataset = "unknown"
    for d in ("inspect","merlin","mimic"):
        if f"{d.lower()}_" in base.lower():
            dataset = d
            break
    prompt = "Long" if "long" in base else "Short" if "short" in base else "Unknown"
    if "alter" in base: edit_file = "alter"
    elif "missing" in base: edit_file = "missing"
    else: edit_file = "unknown"
    return dataset, prompt, edit_file

def standardize_edit(x: str) -> str:
    x = x.lower()
    if x in ("missing_facts","missing"): return "missing"
    if x in ("alter_terms","alter"): return "alter"
    return x

def standardize_summarizer(s: str) -> str:
    s = s.lower()
    # strip common training suffixes that sometimes get glued onto summarizer
    s = re.sub(r"_(llama3|instruct|chat|it)$", "", s)
    # keep your existing aliases
    s = s.replace("llama_3_3_instruct","llama_3_3").replace("llama_3_1_instruct","llama_3_1")
    s = s.replace("gpt-5o","gpt_5o").replace("gpt-5","gpt_5")
    s = s.replace("claude-3.5-sonnet","claude_3_5_sonnet").replace("claude-3-opus","claude_3_opus")
    s = s.replace("llama3_3", "llama_3_3")
    s = s.replace("llama3_1", "llama_3_1")
    return s

def build_tidy_from_csv(fp: str) -> pd.DataFrame:
    dataset, prompt, edit_file = parse_meta_from_filename(fp)
    df = pd.read_csv(fp)
    rows = []
    for col in df.columns:
        m = COL_RE.match(col)
        if not m: 
            continue
        metric = METRIC_RAW_TO_NICE.get(m.group("metric").lower(), m.group("metric"))
        edit = standardize_edit(m.group("edit"))
        diff = m.group("difficulty").lower()
        summarizer = standardize_summarizer(m.group("summarizer"))
        
        # For missing facts I will keep the easy and hard edits only
        if edit == "missing" and diff == "medium":
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        for v in vals:
            if pd.isna(v): 
                continue
            rows.append({
                "dataset": dataset,
                "prompt": prompt.capitalize(),
                "edit_type": edit,
                "metric": metric,
                "difficulty": diff,
                "summarizer": summarizer,
                "score": float(v),
            })
    tidy = pd.DataFrame(rows)
    if not tidy.empty:
        tidy["difficulty"] = pd.Categorical(tidy["difficulty"], ["easy","medium","hard"], ordered=True)
        tidy["edit_type"] = tidy["edit_type"].map(
            {"alter":"Alter Term","missing":"Missing Fact"}
        ).fillna(tidy["edit_type"])
    return tidy

# ======================================================================================
# Statistics
# (verbatim from llm_as_judge_plotting.py)
# ======================================================================================
def wilson_ci(k: int, n: int) -> Tuple[float,float]:
    if n == 0: return (np.nan, np.nan)
    z = 1.959963984540054
    phat = k/n
    denom = 1 + z**2/n
    center = (phat + z**2/(2*n)) / denom
    margin = (z * math.sqrt((phat*(1-phat)/n) + (z**2/(4*n**2)))) / denom
    return (center - margin, center + margin)

def _wilson_ci(k: int, n: int):
    if n == 0: return (np.nan, np.nan)
    z = 1.959963984540054
    phat = k/n
    denom = 1 + z**2/n
    center = (phat + z**2/(2*n)) / denom
    margin = (z * math.sqrt((phat*(1-phat)/n) + (z**2/(4*n**2)))) / denom
    return (center - margin, center + margin)

def _agg_binary_wilson(df_bin: pd.DataFrame, group_cols):
    """Aggregate mean + Wilson CI + raw counts for 0/1 scores."""
    g = (
        df_bin.groupby(group_cols, dropna=False)["score"]
        .agg(mean="mean", k="sum", n="count")
        .reset_index()
    )
    cis = g.apply(lambda r: _wilson_ci(int(r["k"]), int(r["n"])), axis=1)
    g["lo"] = [c[0] for c in cis]; g["hi"] = [c[1] for c in cis]
    return g

def _prep_points(tidy: pd.DataFrame, group_cols):
    """Build merged x/y table for Factual (x) vs Certainty (y) using same grouping."""
    fact = _agg_binary_wilson(tidy[tidy["metric"]=="Factual Faithfulness"], group_cols)
    cert = _agg_binary_wilson(tidy[tidy["metric"]=="Certainty Fidelity"],   group_cols)
    m = pd.merge(fact, cert, on=group_cols, suffixes=("_x","_y"), how="inner")
    # rename + clip to [0,1]
    m = m.rename(columns={
        "mean_x":"x", "lo_x":"x_lo", "hi_x":"x_hi", "k_x":"k_x", "n_x":"n_x",
        "mean_y":"y", "lo_y":"y_lo", "hi_y":"y_hi", "k_y":"k_y", "n_y":"n_y"
    })
    for c in ["x","x_lo","x_hi","y","y_lo","y_hi"]:
        m[c] = m[c].clip(0, 1)
    
    # --- ADD: Bernoulli SD (per-example variability), not what error bars show ---
    # Use p = k/n; SD = sqrt(p(1-p)) to avoid needing raw rows again.
    m["x_std"] = np.sqrt((m["k_x"] / m["n_x"]) * (1.0 - (m["k_x"] / m["n_x"])))
    m["y_std"] = np.sqrt((m["k_y"] / m["n_y"]) * (1.0 - (m["k_y"] / m["n_y"])))
    return m

def _add_all_datasets(points: pd.DataFrame, pool_cols):
    """
    Pool counts across datasets for each tuple in pool_cols, recompute
    means + Wilson CI from pooled k/n. Returns original points + pooled rows.
    """
    agg = (
        points.groupby(pool_cols, as_index=False)
        .agg(k_x=("k_x","sum"), n_x=("n_x","sum"),
             k_y=("k_y","sum"), n_y=("n_y","sum"))
    )
    xs = agg.apply(lambda r: _wilson_ci(int(r["k_x"]), int(r["n_x"])), axis=1)
    ys = agg.apply(lambda r: _wilson_ci(int(r["k_y"]), int(r["n_y"])), axis=1)
    agg["x"] = agg["k_x"]/agg["n_x"]; agg["y"] = agg["k_y"]/agg["n_y"]
    agg["x_lo"] = [c[0] for c in xs]; agg["x_hi"] = [c[1] for c in xs]
    agg["y_lo"] = [c[0] for c in ys]; agg["y_hi"] = [c[1] for c in ys]
    for c in ["x","x_lo","x_hi","y","y_lo","y_hi"]:
        agg[c] = agg[c].clip(0, 1)
    agg["dataset"] = "AllDatasets"
    # Align columns with original (+ any extra group cols)
    keep = ["dataset"] + pool_cols + ["x","x_lo","x_hi","y","y_lo","y_hi","k_x","n_x","k_y","n_y"]
    return pd.concat([points, agg[keep]], ignore_index=True)

def _csv_path_next_to_plot(plot_path: str, suffix: str) -> str:
    base, _ = os.path.splitext(plot_path)
    return f"{base}{suffix}.csv"

def _save_csv_next_to_plot(df: pd.DataFrame, plot_path: str, suffix: str) -> str:
    csv_path = _csv_path_next_to_plot(plot_path, suffix)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    return csv_path

def _export_factual_vs_certainty_scatter_stats_from_points(
    points: pd.DataFrame,
    plot_path: str,
    *,
    dataset: str,
    edit_type: str,
    difficulty: Optional[str] = None,
    suffix: str = "_scatter_points_stats",
) -> None:
    df = points.copy()
    df = df[(df["dataset"] == dataset) & (df["edit_type"] == edit_type)]
    if difficulty is not None:
        df = df[df["difficulty"] == difficulty]
    if df.empty:
        return

    out = df[[
        "dataset",
        *([ "difficulty" ] if "difficulty" in df.columns else []),
        "edit_type",
        "summarizer",
        "prompt",
        "x", "x_lo", "x_hi", "x_std", "n_x", "k_x",
        "y", "y_lo", "y_hi", "y_std", "n_y", "k_y",
    ]].copy()

    # rename to your requested column names
    out = out.rename(columns={
        "x": "x_mean", "x_lo": "x_ci_lo", "x_hi": "x_ci_hi",
        "y": "y_mean", "y_lo": "y_ci_lo", "y_hi": "y_ci_hi",
    })

    _save_csv_next_to_plot(out, plot_path, suffix)

def _wilson_ci_pct(k: int, n: int) -> Tuple[float, float]:
    """Wilson 95% CI for a proportion, returned as percentages (0..100)."""
    if n == 0:
        return (np.nan, np.nan)
    z = 1.959963984540054
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2/(2*n)) / denom
    margin = (z * np.sqrt((phat*(1-phat)/n) + (z**2/(4*n**2)))) / denom
    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    return (lo*100.0, hi*100.0)

def _compute_hallucination_rates_by_prompt(
    df: pd.DataFrame,
    edit_types: List[str],
    summarizers: List[str],
    prompt_order: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    For each edit_type, compute hallucination rate (%) per (summarizer, prompt).
    Hallucinated when score==0.
    Returns {edit_type -> DataFrame[summarizer, prompt, rate_pct, lo_pct, hi_pct, n, k]}.
    """
    if prompt_order is None:
        prompt_order = ["Long", "Short"]

    out = {}
    for ed in edit_types:
        chunk = df[(df["metric"] == "Hallucination") & (df["edit_type"] == ed)].copy()
        rows = []
        for s in summarizers:
            for p in prompt_order:
                g = chunk[(chunk["summarizer"] == s) & (chunk["prompt"] == p)]["score"]
                n = int(g.shape[0])
                if n == 0:
                    rows.append([s, p, np.nan, np.nan, np.nan, 0, 0])
                    continue
                k = int((g == 0).sum())  # hallucinated when score==0
                rate = 100.0 * k / n
                lo, hi = _wilson_ci_pct(k, n)
                rows.append([s, p, rate, lo, hi, n, k])

        tab = pd.DataFrame(
            rows,
            columns=["summarizer", "prompt", "rate_pct", "lo_pct", "hi_pct", "n", "k"]
        )
        out[ed] = tab
    return out

def _perm_test_diff_in_proportions(
    x_long: np.ndarray,
    x_short: np.ndarray,
    B: int = 20000,
    seed: int = 123,
) -> float:
    """
    Two-sided permutation test for difference in means between two 0/1 arrays.
    Returns p-value. Uses pooled-label permutation.
    """
    x_long = np.asarray(x_long, dtype=float)
    x_short = np.asarray(x_short, dtype=float)
    x_long = x_long[~np.isnan(x_long)]
    x_short = x_short[~np.isnan(x_short)]

    if x_long.size == 0 or x_short.size == 0:
        return np.nan

    obs = float(np.mean(x_long) - np.mean(x_short))
    pooled = np.concatenate([x_long, x_short], axis=0)
    nL = x_long.size

    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(B):
        perm = rng.permutation(pooled)
        d = float(np.mean(perm[:nL]) - np.mean(perm[nL:]))
        if abs(d) >= abs(obs):
            cnt += 1
    # add-one smoothing
    return (cnt + 1.0) / (B + 1.0)

def _hallucination_prompt_stats(
    df: pd.DataFrame,
    edit_types: List[str],
    summarizers: List[str],
    B: int = 20000,
    seed: int = 123,
) -> pd.DataFrame:
    """
    Per (edit_type, summarizer): test Long vs Short difference in hallucination rate.
    Uses permutation test on hallucinated indicator (score==0).
    """
    rows = []
    for ed in edit_types:
        chunk = df[(df["metric"] == "Hallucination") & (df["edit_type"] == ed)].copy()
        if chunk.empty:
            continue
        chunk["hallucinated"] = (chunk["score"] == 0).astype(int)

        for s in summarizers:
            a = chunk[(chunk["summarizer"] == s) & (chunk["prompt"] == "Long")]["hallucinated"].to_numpy()
            b = chunk[(chunk["summarizer"] == s) & (chunk["prompt"] == "Short")]["hallucinated"].to_numpy()

            n_long = int(a.size); n_short = int(b.size)
            k_long = int(a.sum()) if n_long else 0
            k_short = int(b.sum()) if n_short else 0

            rate_long = (100.0 * k_long / n_long) if n_long else np.nan
            rate_short = (100.0 * k_short / n_short) if n_short else np.nan
            diff = (rate_long - rate_short) if (n_long and n_short) else np.nan

            p = _perm_test_diff_in_proportions(a, b, B=B, seed=seed) if (n_long and n_short) else np.nan

            rows.append({
                "edit_type": ed,
                "summarizer": s,
                "n_long": n_long, "k_long": k_long, "rate_long_pct": rate_long,
                "n_short": n_short, "k_short": k_short, "rate_short_pct": rate_short,
                "diff_long_minus_short_pct": diff,
                "p_value_perm": p,
                "B": B,
            })
    return pd.DataFrame(rows)

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
# Driver (new; no plotting)
# ==========================================================================================
def factual_faithfulness_by_prompt(tidy: pd.DataFrame, edit_type: str, out_dir: str) -> None:
    """Figure 3b numbers. Statistical lines are those of plot_factual_bar_by_summarizer_prompt."""
    df = tidy[tidy["metric"] == "Factual Faithfulness"].copy()
    if edit_type is not None:
        df = df[df["edit_type"] == edit_type]
    if df.empty:
        return
    base = os.path.join(out_dir, f"FactualFaithfulness_AllDatasets_{edit_type.replace(' ','')}_SummarizerPrompt")

    # mean + Wilson CI per (summarizer, prompt)
    g = (
        df.groupby(["summarizer", "prompt"], dropna=False)["score"]
        .agg(mean="mean", k="sum", n="count", std="std")
        .reset_index()
    )
    cis = g.apply(lambda r: wilson_ci(int(r["k"]), int(r["n"])), axis=1)
    g["ci_lo"] = [c[0] for c in cis]
    g["ci_hi"] = [c[1] for c in cis]
    g.insert(0, "metric", "Factual Faithfulness")
    g.insert(0, "edit_type", edit_type if edit_type is not None else "")
    g.insert(0, "dataset", "AllDatasets")
    _save_csv_next_to_plot(g, base, "_bar_stats")

    # Long vs Short permutation test per summarizer (same call as in the figure script)
    stats = _agg_binary_wilson(df, group_cols=["summarizer", "prompt"])
    total = stats.groupby("summarizer").agg(k=("k", "sum"), n=("n", "sum")).reset_index()
    total = total[total["n"] > 0]
    total["mean"] = total["k"] / total["n"]
    ordered_summarizers = total.sort_values("mean", ascending=True)["summarizer"].tolist()
    rows = []
    for s in ordered_summarizers:
        a = df[(df["summarizer"] == s) & (df["prompt"] == "Long")]["score"].to_numpy()
        b = df[(df["summarizer"] == s) & (df["prompt"] == "Short")]["score"].to_numpy()
        if a.size == 0 or b.size == 0:
            continue
        pval = _perm_test_diff_in_proportions(a, b, B=20000, seed=123)
        rows.append({
            "dataset": "AllDatasets", "edit_type": edit_type, "summarizer": s,
            "n_long": int(a.size), "mean_long": float(np.mean(a)),
            "n_short": int(b.size), "mean_short": float(np.mean(b)),
            "diff_long_minus_short": float(np.mean(a) - np.mean(b)),
            "p_value_perm": float(pval), "label": _p_to_stars(float(pval)), "B": 20000,
        })
    _save_csv_next_to_plot(pd.DataFrame(rows), base, "_long_vs_short_perm_tests")


def faithfulness_vs_certainty(tidy: pd.DataFrame, out_dir: str) -> None:
    """Figure 3a numbers. Mirrors the export loop of make_faith_vs_cert_plots."""
    group_cols = ["dataset", "edit_type", "summarizer", "prompt"]
    points = _prep_points(tidy, group_cols)
    points = _add_all_datasets(points, pool_cols=["edit_type", "summarizer", "prompt"])
    for ds in sorted(points["dataset"].unique()):
        for ed in sorted(points["edit_type"].unique()):
            sub = points[(points["dataset"] == ds) & (points["edit_type"] == ed)]
            if sub.empty:
                continue
            base = os.path.join(out_dir, f"{ds}_{str(ed).replace(' ','')}_Factual_vs_Certainty")
            _export_factual_vs_certainty_scatter_stats_from_points(
                points, base, dataset=ds, edit_type=ed, difficulty=None)


def hallucination_rates(tidy: pd.DataFrame, judge: str, global_summarizers: List[str], out_dir: str) -> None:
    """Figure 4b numbers. Mirrors the exports of plot_hallucination_rates_by_judge (one judge)."""
    edit_types = ["Alter Term", "Missing Fact"]
    prompt_order = [p for p in ["Long", "Short"] if p in set(tidy["prompt"].dropna().unique())]
    base = os.path.join(out_dir, f"HallucinationRates_AllDatasets_{judge}")

    rates = _compute_hallucination_rates_by_prompt(tidy, edit_types, global_summarizers, prompt_order=prompt_order)
    exports = []
    for ed in edit_types:
        exp = rates[ed].copy()
        exp.insert(0, "judge", judge)
        exp.insert(1, "edit_type", ed)
        exp["dataset"] = "AllDatasets"
        exports.append(exp)
    _save_csv_next_to_plot(pd.concat(exports, ignore_index=True), base, "_rates_by_prompt")

    if set(["Long", "Short"]).issubset(set(prompt_order)):
        tests = _hallucination_prompt_stats(tidy, edit_types, global_summarizers, B=20000, seed=123)
        tests.insert(0, "judge", judge)
        tests.insert(0, "dataset", "AllDatasets")
        _save_csv_next_to_plot(tests, base, "_long_vs_short_perm_tests")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores-dir", required=True,
                    help="Folder with <judge>_<dataset>_<long|short>_<alter_terms|missing_facts>_combined_cleaned_wide.csv")
    ap.add_argument("--judge", default="gpt_5", help="Judge prefix of the score files (gpt_5 or medgemma)")
    ap.add_argument("--out-dir", required=True, help="Output folder for the tables")
    ap.add_argument("--file-order", default=None,
                    help="Optional text file listing the score-file names in processing order. The permutation "
                         "p-values depend on row order through the random-number stream, so use "
                         "analysis/file_order_<judge>.txt to reproduce the published p-values exactly. "
                         "Without it files are sorted by name and p-values differ only by Monte Carlo error.")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.file_order:
        with open(args.file_order) as fh:
            names = [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]
        files = [os.path.join(args.scores_dir, n) for n in names]
        missing = [f for f in files if not os.path.exists(f)]
        if missing:
            raise SystemExit(f"Files listed in {args.file_order} not found: {missing}")
    else:
        files = sorted(glob.glob(os.path.join(args.scores_dir, f"{args.judge}_*_combined_cleaned_wide.csv")))
    if not files:
        raise SystemExit(f"No score files matching {args.judge}_*_combined_cleaned_wide.csv in {args.scores_dir}")
    print(f"[INFO] {len(files)} score files")
    tidy = pd.concat([t for t in (build_tidy_from_csv(fp) for fp in files) if t is not None and not t.empty],
                     ignore_index=True)
    tidy.to_csv(os.path.join(args.out_dir, f"all_datasets_tidy_{args.judge}.csv"), index=False)
    global_summarizers = sorted(tidy["summarizer"].dropna().unique().tolist())

    for ed in sorted(tidy["edit_type"].unique()):
        factual_faithfulness_by_prompt(tidy, ed, args.out_dir)
    faithfulness_vs_certainty(tidy, args.out_dir)
    hallucination_rates(tidy, args.judge, global_summarizers, args.out_dir)
    print(f"[OK] tables written to {args.out_dir}")


if __name__ == "__main__":
    main()
