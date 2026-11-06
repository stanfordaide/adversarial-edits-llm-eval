#!/usr/bin/env python3
r"""
Concordance between the published permutation tests and the new model-based tests.

Purpose: check that the multi-factor models (fit_models.py) are CONFIRMATORY - they
reproduce the inferences of the permutation tests reported in the paper - while
additionally supplying effect sizes, confidence intervals and formal interaction tests,
and while accounting for the repeated measures on each report that the permutation
tests ignore.

Permutation results used (outputs of the analysis/ scripts, same values as in the paper):
  <results-dir>/HallucinationRates_AllDatasets_gpt_5_long_vs_short_perm_tests.csv
      -> Detailed vs Concise hallucination rate, per summarizer x edit type (B=20,000)
  <results-dir>/AllDatasets_SensitivityDistribution_{AlterTerm,MissingFact}_long_vs_short_perm_tests.csv
      -> Detailed vs Concise 2x3 distribution of the sensitivity score (B=20,000)

Model results used:
  <model-dir>/contrasts_prompt_by_summarizer.csv  (from fit_models.py)

Note on comparability: the published sensitivity test is an omnibus test on the
full 0/1/2 distribution, whereas the model contrast is a single proportional-odds
shift. They test related but not identical hypotheses, so exact p-value equality is
not expected; the comparison is of the inferential conclusion and its direction.

Usage:
    python analysis/multifactor_models/concordance_permutation.py \
        --results-dir results/gpt_5 --model-dir results/multifactor_models
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

# Set from the command line in main().
RESULTS_DIR = "results/gpt_5"                 # outputs of analysis/summary_statistics.py + sensitivity_statistics.py
OUT_DIR = "results/multifactor_models"        # outputs of fit_models.py; concordance tables are written here too
PERM_HALL = PERM_SENS = MODEL_CONTRASTS = DATA = None


def _configure(results_dir: str, model_dir: str) -> None:
    global RESULTS_DIR, OUT_DIR, PERM_HALL, PERM_SENS, MODEL_CONTRASTS, DATA
    RESULTS_DIR, OUT_DIR = results_dir, model_dir
    PERM_HALL = os.path.join(RESULTS_DIR, "HallucinationRates_AllDatasets_gpt_5_long_vs_short_perm_tests.csv")
    PERM_SENS = {
        "Alter Term": os.path.join(RESULTS_DIR, "AllDatasets_SensitivityDistribution_AlterTerm_long_vs_short_perm_tests.csv"),
        "Missing Fact": os.path.join(RESULTS_DIR, "AllDatasets_SensitivityDistribution_MissingFact_long_vs_short_perm_tests.csv"),
    }
    MODEL_CONTRASTS = os.path.join(OUT_DIR, "contrasts_prompt_by_summarizer.csv")
    DATA = os.path.join(OUT_DIR, "model_dataset_gpt5_judge.csv")


SUMMARIZER_LABEL = {
    "claude": "Claude Sonnet 4.5", "gpt": "GPT-5-mini", "llama_3_1": "Llama 3.1 (8B)",
    "llama_3_3": "Llama 3.3 (70B)", "medgemma": "MedGemma 27B", "qwen": "Qwen 3 (4B)",
}
ALPHA = 0.05


def observed_prompt_deltas(outcome: str) -> pd.DataFrame:
    """Observed Detailed - Concise difference in mean outcome, per summarizer x edit type."""
    df = pd.read_csv(DATA)
    df = df[df["outcome_name"] == outcome]
    g = df.groupby(["summarizer_label", "edit_type", "prompt_label"])["outcome"].mean().unstack()
    out = (g["Detailed"] - g["Concise"]).reset_index(name="observed_delta_detailed_minus_concise")
    out.insert(0, "outcome", outcome)
    return out


def load_permutation() -> pd.DataFrame:
    rows = []

    hall = pd.read_csv(PERM_HALL)
    hall = hall[hall["judge"] == "gpt_5"]
    for _, r in hall.iterrows():
        rows.append({
            "outcome": "Hallucination (present)",
            "summarizer_label": SUMMARIZER_LABEL[r["summarizer"]],
            "edit_type": r["edit_type"],
            "perm_test": "difference in hallucination rate (Detailed vs Concise)",
            "perm_effect": r["diff_long_minus_short_pct"] / 100.0,
            "perm_effect_units": "proportion (Detailed - Concise)",
            "perm_p_value": r["p_value_perm"],
            "perm_B": int(r["B"]),
        })

    for edit, path in PERM_SENS.items():
        s = pd.read_csv(path)
        for _, r in s.iterrows():
            long_c = np.array([r["long_c0"], r["long_c1"], r["long_c2"]], dtype=float)
            short_c = np.array([r["short_c0"], r["short_c1"], r["short_c2"]], dtype=float)
            mean_long = (long_c * np.arange(3)).sum() / long_c.sum()
            mean_short = (short_c * np.arange(3)).sum() / short_c.sum()
            rows.append({
                "outcome": "Sensitivity to Changes",
                "summarizer_label": SUMMARIZER_LABEL[r["summarizer"]],
                "edit_type": edit,
                "perm_test": "2x3 distribution of sensitivity score (Detailed vs Concise)",
                "perm_effect": mean_long - mean_short,
                "perm_effect_units": "mean 0-2 score (Detailed - Concise)",
                "perm_p_value": r["p_value_perm_chi2_2x3"],
                "perm_B": int(r["B"]),
            })

    return pd.DataFrame(rows)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results/gpt_5",
                    help="Folder with the permutation-test tables from analysis/summary_statistics.py and sensitivity_statistics.py")
    ap.add_argument("--model-dir", default="results/multifactor_models", help="Folder with the fit_models.py outputs")
    args = ap.parse_args()
    _configure(args.results_dir, args.model_dir)
    if not os.path.exists(MODEL_CONTRASTS):
        print(f"[ERROR] {MODEL_CONTRASTS} not found - run fit_models.py first.")
        return 1

    perm = load_permutation()
    model = pd.read_csv(MODEL_CONTRASTS)

    mg = perm.merge(
        model[["outcome", "summarizer_label", "edit_type", "log_or", "odds_ratio",
               "or_ci_low", "or_ci_high", "p_value", "p_holm_within_outcome"]],
        on=["outcome", "summarizer_label", "edit_type"], how="left", validate="one_to_one",
    ).rename(columns={
        "p_value": "model_p_value",
        "p_holm_within_outcome": "model_p_holm",
        "log_or": "model_log_or",
        "odds_ratio": "model_odds_ratio",
        "or_ci_low": "model_or_ci_low",
        "or_ci_high": "model_or_ci_high",
    })
    assert mg["model_p_value"].notna().all(), "some permutation comparisons had no model counterpart"

    # observed deltas, to check that the model contrast points the same way
    obs = pd.concat([observed_prompt_deltas(o) for o in mg["outcome"].unique()], ignore_index=True)
    n_before = len(mg)
    mg = mg.merge(obs, on=["outcome", "summarizer_label", "edit_type"],
                  how="left", validate="one_to_one")
    assert len(mg) == n_before, "observed-delta merge changed the row count"

    mg["perm_significant"] = mg["perm_p_value"] < ALPHA
    mg["model_significant"] = mg["model_p_value"] < ALPHA
    mg["model_significant_holm"] = mg["model_p_holm"] < ALPHA
    mg["agreement"] = np.where(
        mg["perm_significant"] == mg["model_significant"],
        np.where(mg["perm_significant"], "both significant", "both non-significant"),
        np.where(mg["model_significant"], "model only", "permutation only"),
    )
    # Direction check. The permutation test compares raw pooled proportions, the model
    # contrast is adjusted for dataset and edit-intensity composition, so the two can
    # point opposite ways when the effect is essentially null. Only a disagreement on a
    # non-negligible effect is meaningful, so both checks are reported.
    mg["direction_agrees"] = (
        np.sign(mg["model_log_or"]) == np.sign(mg["observed_delta_detailed_minus_concise"])
    ) | np.isclose(mg["observed_delta_detailed_minus_concise"], 0)
    mg["effect_is_negligible"] = (mg["model_odds_ratio"].sub(1).abs() < 0.05) & (
        mg["observed_delta_detailed_minus_concise"].abs() < 0.02
    )
    mg["direction_conflict_nonnegligible"] = (~mg["direction_agrees"]) & (~mg["effect_is_negligible"])

    mg = mg.sort_values(["outcome", "edit_type", "summarizer_label"]).reset_index(drop=True)
    path = os.path.join(OUT_DIR, "concordance_permutation_vs_model.csv")
    mg.to_csv(path, index=False)

    # ---- summary -------------------------------------------------------------------
    n = len(mg)
    conc = int((mg["perm_significant"] == mg["model_significant"]).sum())
    both_sig = int(((mg["perm_significant"]) & (mg["model_significant"])).sum())
    both_ns = int(((~mg["perm_significant"]) & (~mg["model_significant"])).sum())
    model_only = int(((~mg["perm_significant"]) & (mg["model_significant"])).sum())
    perm_only = int(((mg["perm_significant"]) & (~mg["model_significant"])).sum())
    rho, prho = stats.spearmanr(-np.log10(mg["perm_p_value"]), -np.log10(mg["model_p_value"]))

    summary = pd.DataFrame([{
        "n_comparisons": n,
        "concordant_at_alpha_0.05": conc,
        "concordance_pct": 100.0 * conc / n,
        "both_significant": both_sig,
        "both_non_significant": both_ns,
        "model_significant_only": model_only,
        "permutation_significant_only": perm_only,
        "direction_agreement": int(mg["direction_agrees"].sum()),
        "direction_conflicts_on_nonnegligible_effects": int(mg["direction_conflict_nonnegligible"].sum()),
        "spearman_rho_neglog10_p": float(rho),
        "spearman_p": float(prho),
        "model_significant_after_holm": int(mg["model_significant_holm"].sum()),
        "permutation_significant": int(mg["perm_significant"].sum()),
    }])
    spath = os.path.join(OUT_DIR, "concordance_summary.csv")
    summary.to_csv(spath, index=False)

    print(f"[OK] {path}")
    print(f"[OK] {spath}\n")
    print("=== Concordance: published permutation tests vs new model-based tests ===\n")
    show = mg[["outcome", "edit_type", "summarizer_label", "perm_p_value", "model_p_value",
               "model_odds_ratio", "model_or_ci_low", "model_or_ci_high", "agreement"]]
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(show.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("\n=== Summary ===")
    print(summary.T.to_string(header=False))
    print(f"\nDirection of effect agrees with the observed difference in "
          f"{int(mg['direction_agrees'].sum())}/{n} comparisons; "
          f"{int(mg['direction_conflict_nonnegligible'].sum())} conflict on a "
          f"non-negligible effect.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
