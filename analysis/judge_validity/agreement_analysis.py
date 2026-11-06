#!/usr/bin/env python3
r"""
Judge-validity reanalysis of the radiologist reader study (n = 105).

Motivation
    With skewed label prevalence, Cohen's kappa can be low despite high observed agreement
    (the prevalence paradox), while macro-F1 applies no chance correction at all.
    This script quantifies judge-radiologist agreement with prevalence-robust,
    chance-corrected statistics and tests whether the GPT-5 judge agrees with each
    radiologist about as well as the two radiologists agree with each other.

What this script computes (the reader-study CSV is only read, never written):

  1. Prevalence-robust, chance-corrected agreement
        Gwet's AC1 (binary metrics) / AC2 with quadratic weights (ordinal sensitivity),
        with Brennan-Prediger / PABAK as a second prevalence-robust coefficient and
        Cohen's kappa for transparency. Case-resampling bootstrap 95% CIs.
  2. A direct test of whether agreement beats chance
        Permutation test: one rater's labels are permuted across cases, which destroys
        any case-level association while keeping BOTH raters' marginal distributions
        fixed. Because the marginals are fixed, kappa, Scott's pi, AC1 and PABAK are all
        monotone in observed agreement, so this single p-value answers "better than
        chance?" for every coefficient at once. The permutation mean of each coefficient
        is also reported: it is the value that coefficient takes under pure chance given
        the observed prevalences. For AC1 this is NOT zero when prevalence is skewed, so
        AC1 on its own does not establish that agreement beats chance.
  3. The paper's judge-vs-radiologist comparison, tested formally
        Paired bootstrap of  agreement(GPT-5, radiologist R) - agreement(other radiologist, R)
        for each reference radiologist, and pooled, for every coefficient.
  4. Validity against a cleaner reference standard
        On the cases where the two radiologists agree, GPT-5 is compared with their
        consensus label: accuracy vs the majority-class baseline, balanced accuracy
        (chance = 0.5, prevalence-robust), detection of the problem class, and AC1/kappa.
  5. Systematic calibration differences between raters
        Marginal prevalences, McNemar (binary) and Stuart-Maxwell (ordinal) tests.

Verification performed before any new statistic is reported
    - Cohen's kappa and symmetric macro-F1 are checked against scikit-learn;
    - binary AC1 and PABAK are checked against their closed forms;
    - AC1/AC2 and Brennan-Prediger are checked against the reference package `irrCAC`
      when it is importable (optional).

Reader-study data: all 105 cases are GPT-5-mini summaries generated with the concise
prompt; every rater, including GPT-5, codes hallucination as 1 = NO hallucination.
Radiologist columns are expected as <metric>_rad_a / <metric>_rad_b (hallucination:
hallucination_rate_rad_a / hallucination_rate_rad_b).

Usage
    python analysis/judge_validity/agreement_analysis.py \
        --reader-csv data/reader_study/reader_study_ratings.csv --out-dir results/judge_validity
    (optional: PYTHONPATH=<dir containing irrCAC> for the external cross-check)
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score
from statsmodels.stats.contingency_tables import SquareTable, mcnemar

# Set from the command line in main().
READER_CSV = "data/reader_study/reader_study_ratings.csv"
OUT_DIR = "results/judge_validity"
LOG_PATH = os.path.join(OUT_DIR, "analysis_log.txt")

SEED = 20260918
B_BOOT = 5000
B_PERM = 10000

METRICS = ["factual_faithfulness", "certainty_fidelity", "sensitivity_to_changes", "hallucination"]
METRIC_LABEL = {
    "factual_faithfulness": "Factual Faithfulness",
    "certainty_fidelity": "Certainty Fidelity",
    "sensitivity_to_changes": "Sensitivity to Changes",
    "hallucination": "Hallucination",
}
N_CATEGORIES = {"factual_faithfulness": 2, "certainty_fidelity": 2,
                "sensitivity_to_changes": 3, "hallucination": 2}
# Quadratic weighting for the ordinal metric matches the published weighted kappa.
PRIMARY_WEIGHTS = {"factual_faithfulness": "identity", "certainty_fidelity": "identity",
                   "sensitivity_to_changes": "quadratic", "hallucination": "identity"}
# The "problem" category for each metric (the outcome the judge exists to catch).
PROBLEM_CLASS = {"factual_faithfulness": 0, "certainty_fidelity": 0, "hallucination": 0}

# Source columns. Radiologist identities are mapped to anonymous labels immediately.
COLUMNS = {
    "gpt_5": {
        "factual_faithfulness": "factual_faithfulness_gpt_5",
        "sensitivity_to_changes": "sensitivity_to_changes_gpt_5",
        "certainty_fidelity": "certainty_fidelity_gpt_5",
        "hallucination": "hallucination_gpt_5",
    },
    "rad_a": {
        "factual_faithfulness": "factual_faithfulness_rad_a",
        "sensitivity_to_changes": "sensitivity_to_changes_rad_a",
        "certainty_fidelity": "certainty_fidelity_rad_a",
        "hallucination": "hallucination_rate_rad_a",
    },
    "rad_b": {
        "factual_faithfulness": "factual_faithfulness_rad_b",
        "sensitivity_to_changes": "sensitivity_to_changes_rad_b",
        "certainty_fidelity": "certainty_fidelity_rad_b",
        "hallucination": "hallucination_rate_rad_b",
    },
    "medgemma": {
        "factual_faithfulness": "factual_faithfullness_medgemma",   # source typo
        "sensitivity_to_changes": "sensitivity_to_changes_medgemma",
        "certainty_fidelity": "certainty_fidelity_medgemma",
        "hallucination": "hallucination_medgemma",
    },
}
RATER_LABEL = {"gpt_5": "GPT-5 judge", "rad_a": "Radiologist A",
               "rad_b": "Radiologist B", "medgemma": "MedGemma judge"}

PRIMARY_PAIRS = [("gpt_5", "rad_a"), ("gpt_5", "rad_b"), ("rad_a", "rad_b")]
SECONDARY_PAIRS = [("gpt_5", "medgemma"), ("medgemma", "rad_a"), ("medgemma", "rad_b")]
ALL_PAIRS = PRIMARY_PAIRS + SECONDARY_PAIRS
COEFS = ["percent_agreement", "kappa", "gwet_ac", "brennan_prediger", "scott_pi", "macro_f1"]

_log: list[str] = []


def log(msg: str = "") -> None:
    print(msg)
    _log.append(msg)


def pair_label(p) -> str:
    return f"{RATER_LABEL[p[0]]} vs {RATER_LABEL[p[1]]}"


# ======================================================================================
# coefficients (two raters, categories coded 0..q-1)
# ======================================================================================
def weight_matrix(q: int, kind: str) -> np.ndarray:
    k = np.arange(q)
    if kind == "identity":
        return np.eye(q)
    if kind == "quadratic":
        return 1.0 - ((k[:, None] - k[None, :]) / (q - 1)) ** 2
    if kind == "linear":
        return 1.0 - np.abs(k[:, None] - k[None, :]) / (q - 1)
    raise ValueError(kind)


def _ratio(po: float, pe: float) -> float:
    return (po - pe) / (1.0 - pe) if pe < 1.0 - 1e-12 else np.nan


def coefficients(x: np.ndarray, y: np.ndarray, q: int, W: np.ndarray) -> dict:
    """
    All agreement coefficients from one joint table.

      percent_agreement   exact (unweighted) agreement, as published
      kappa               Cohen: pe = sum_kl w_kl p1_k p2_l       (rater-specific marginals)
      scott_pi            Scott: pe = sum_kl w_kl pi_k pi_l        (pooled marginals)
      gwet_ac             Gwet AC1/AC2: pe = T_w/(q(q-1)) * sum_k pi_k(1-pi_k)
      brennan_prediger    pe = T_w / q^2   (= PABAK for binary, identity weights)
      macro_f1            mean over present classes of 2 n_kk / (n1_k + n2_k)
                          (symmetric in the two raters; equals sklearn's macro F1)
    """
    n = x.size
    P = np.bincount(x * q + y, minlength=q * q).reshape(q, q) / n
    p1, p2 = P.sum(axis=1), P.sum(axis=0)
    pi = (p1 + p2) / 2.0
    po_w = float((W * P).sum())
    Tw = float(W.sum())
    present = (p1 + p2) > 0
    diag = np.diag(P)
    f1 = 2.0 * diag[present] / (p1[present] + p2[present])
    return {
        "percent_agreement": float(np.trace(P)),
        "weighted_agreement": po_w,
        "kappa": _ratio(po_w, float(p1 @ W @ p2)),
        "scott_pi": _ratio(po_w, float(pi @ W @ pi)),
        "gwet_ac": _ratio(po_w, Tw / (q * (q - 1)) * float((pi * (1 - pi)).sum())),
        "brennan_prediger": _ratio(po_w, Tw / q ** 2),
        "macro_f1": float(f1.mean()) if f1.size else np.nan,
    }


# ======================================================================================
# verification
# ======================================================================================
def verify(df: pd.DataFrame, labels: dict) -> list[str]:
    msgs = []

    # (2) kappa + macro-F1 against sklearn on the full data
    dev_k = dev_f = 0.0
    for m in METRICS:
        q = N_CATEGORIES[m]
        W = weight_matrix(q, PRIMARY_WEIGHTS[m])
        for a, b in ALL_PAIRS:
            x, y = labels[a][m], labels[b][m]
            c = coefficients(x, y, q, W)
            sk = cohen_kappa_score(x, y, labels=list(range(q)),
                                   weights="quadratic" if PRIMARY_WEIGHTS[m] == "quadratic" else None)
            dev_k = max(dev_k, abs(c["kappa"] - sk))
            f12 = f1_score(x, y, average="macro", zero_division=0)
            f21 = f1_score(y, x, average="macro", zero_division=0)
            dev_f = max(dev_f, abs(c["macro_f1"] - 0.5 * (f12 + f21)))
    assert dev_k < 1e-9 and dev_f < 1e-9, (dev_k, dev_f)
    msgs.append(f"kappa and symmetric macro-F1 match sklearn (max dev {dev_k:.1e}, {dev_f:.1e})")

    # (4) closed forms for binary AC1 / PABAK
    for m in ["factual_faithfulness", "hallucination"]:
        x, y = labels["gpt_5"][m], labels["rad_a"][m]
        po = float(np.mean(x == y))
        pi1 = (x.mean() + y.mean()) / 2
        ac1_closed = (po - 2 * pi1 * (1 - pi1)) / (1 - 2 * pi1 * (1 - pi1))
        c = coefficients(x, y, 2, np.eye(2))
        assert abs(c["gwet_ac"] - ac1_closed) < 1e-12 and abs(c["brennan_prediger"] - (2 * po - 1)) < 1e-12
    msgs.append("binary AC1 and PABAK match their closed forms "
                "(AC1 = (po - 2pi(1-pi)) / (1 - 2pi(1-pi)); PABAK = 2po - 1)")

    # (5) optional external reference implementation
    try:
        from irrCAC.raw import CAC  # noqa: WPS433
        dev_g = dev_bp = 0.0
        for m in METRICS:
            q = N_CATEGORIES[m]
            wk = PRIMARY_WEIGHTS[m]
            for a, b in ALL_PAIRS:
                x, y = labels[a][m], labels[b][m]
                c = coefficients(x, y, q, weight_matrix(q, wk))
                cac = CAC(pd.DataFrame({"r1": x, "r2": y}), weights=wk, categories=list(range(q)))
                g = cac.gwet()["est"]["coefficient_value"]
                bp = cac.bp()["est"]["coefficient_value"]
                dev_g = max(dev_g, abs(c["gwet_ac"] - g))
                dev_bp = max(dev_bp, abs(c["brennan_prediger"] - bp))
        # irrCAC rounds reported coefficients to 5 decimals
        assert dev_g < 1e-4 and dev_bp < 1e-4, (dev_g, dev_bp)
        msgs.append(f"Gwet AC1/AC2 and Brennan-Prediger match the irrCAC reference package "
                    f"(max dev {dev_g:.1e}, {dev_bp:.1e}; irrCAC reports 5 decimals)")
    except ImportError:
        msgs.append("[skipped] irrCAC not importable - external cross-check not run")
    return msgs


# ======================================================================================
# main analysis
# ======================================================================================
def main() -> int:
    global READER_CSV, OUT_DIR, LOG_PATH
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reader-csv", default=READER_CSV, help="Reader-study ratings (105 cases)")
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()
    READER_CSV, OUT_DIR = args.reader_csv, args.out_dir
    LOG_PATH = os.path.join(OUT_DIR, "analysis_log.txt")
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(READER_CSV)
    assert len(df) == 105 and df["study_id"].is_unique
    labels = {r: {m: df[c].astype(int).to_numpy() for m, c in cols.items()} for r, cols in COLUMNS.items()}
    for r in labels:
        for m in METRICS:
            v = labels[r][m]
            assert v.min() >= 0 and v.max() < N_CATEGORIES[m], (r, m)

    log(f"[DATA] {READER_CSV}")
    log(f"[DATA] n = {len(df)} cases; datasets {df.dataset.value_counts().to_dict()}")
    log("\n[VERIFY]")
    for msg in verify(df, labels):
        log("   " + msg)

    rng = np.random.default_rng(SEED)
    n = len(df)
    boot_idx = rng.integers(0, n, size=(B_BOOT, n))

    # ---------------------------------------------------------------- point + bootstrap
    rows = []
    boot_store: dict = {}
    for m in METRICS:
        q = N_CATEGORIES[m]
        W = weight_matrix(q, PRIMARY_WEIGHTS[m])
        for p in ALL_PAIRS:
            x, y = labels[p[0]][m], labels[p[1]][m]
            point = coefficients(x, y, q, W)
            boot = {k: np.empty(B_BOOT) for k in COEFS}
            for bi in range(B_BOOT):
                ii = boot_idx[bi]
                cb = coefficients(x[ii], y[ii], q, W)
                for k in COEFS:
                    boot[k][bi] = cb[k]
            boot_store[(m, p)] = boot

            # permutation null: permute y across cases (both marginals preserved)
            perm = {k: np.empty(B_PERM) for k in COEFS + ["weighted_agreement"]}
            prng = np.random.default_rng([SEED, METRICS.index(m), ALL_PAIRS.index(p)])
            for bi in range(B_PERM):
                cp = coefficients(x, prng.permutation(y), q, W)
                for k in perm:
                    perm[k][bi] = cp[k]
            rec = {
                "metric": METRIC_LABEL[m], "weights": PRIMARY_WEIGHTS[m],
                "pair": pair_label(p), "pair_type": "primary" if p in PRIMARY_PAIRS else "secondary (MedGemma)",
                "n": n,
            }
            for k in COEFS:
                lo, hi = np.nanpercentile(boot[k], [2.5, 97.5])
                rec[k] = point[k]
                rec[f"{k}_ci_low"] = lo
                rec[f"{k}_ci_high"] = hi
                rec[f"{k}_chance_expected"] = float(np.nanmean(perm[k]))
            # one-sided permutation p for "agreement exceeds chance" (on weighted agreement,
            # to which kappa / pi / AC / BP are all monotone under fixed marginals)
            wa_obs = point["weighted_agreement"]
            rec["perm_p_beats_chance"] = float(
                (1 + np.sum(perm["weighted_agreement"] >= wa_obs - 1e-12)) / (B_PERM + 1))
            rec["perm_p_macro_f1"] = float((1 + np.sum(perm["macro_f1"] >= point["macro_f1"] - 1e-12)) / (B_PERM + 1))
            rows.append(rec)
    coef_df = pd.DataFrame(rows)
    coef_df.to_csv(os.path.join(OUT_DIR, "agreement_coefficients.csv"), index=False)

    # ---------------------------------------------------------------- overall (mean of 4)
    overall = []
    for p in ALL_PAIRS:
        rec = {"pair": pair_label(p), "pair_type": "primary" if p in PRIMARY_PAIRS else "secondary (MedGemma)"}
        for k in COEFS:
            pts = [coef_df[(coef_df.metric == METRIC_LABEL[m]) & (coef_df.pair == pair_label(p))][k].iloc[0]
                   for m in METRICS]
            bs = np.nanmean(np.vstack([boot_store[(m, p)][k] for m in METRICS]), axis=0)
            rec[k] = float(np.mean(pts))
            rec[f"{k}_ci_low"], rec[f"{k}_ci_high"] = np.nanpercentile(bs, [2.5, 97.5])
        overall.append(rec)
    overall_df = pd.DataFrame(overall)
    overall_df.to_csv(os.path.join(OUT_DIR, "agreement_overall.csv"), index=False)

    # ---------------------------------------------------------------- judge vs inter-rater
    diff_rows = []
    comparisons = [
        ("GPT-5 vs Rad A  minus  Rad B vs Rad A", ("gpt_5", "rad_a"), ("rad_a", "rad_b")),
        ("GPT-5 vs Rad B  minus  Rad A vs Rad B", ("gpt_5", "rad_b"), ("rad_a", "rad_b")),
    ]
    for scope in METRICS + ["overall"]:
        for k in ["kappa", "gwet_ac", "brennan_prediger", "macro_f1"]:
            def get(p):
                if scope == "overall":
                    pts = np.mean([coef_df[(coef_df.metric == METRIC_LABEL[m]) & (coef_df.pair == pair_label(p))][k].iloc[0] for m in METRICS])
                    bs = np.nanmean(np.vstack([boot_store[(m, p)][k] for m in METRICS]), axis=0)
                else:
                    pts = coef_df[(coef_df.metric == METRIC_LABEL[scope]) & (coef_df.pair == pair_label(p))][k].iloc[0]
                    bs = boot_store[(scope, p)][k]
                return float(pts), bs
            ja, bja = get(("gpt_5", "rad_a"))
            jb, bjb = get(("gpt_5", "rad_b"))
            ab, bab = get(("rad_a", "rad_b"))
            for name, est, bs in [
                (comparisons[0][0], ja - ab, bja - bab),
                (comparisons[1][0], jb - ab, bjb - bab),
                ("mean(GPT-5 vs Rad A, GPT-5 vs Rad B)  minus  Rad A vs Rad B", (ja + jb) / 2 - ab, (bja + bjb) / 2 - bab),
            ]:
                lo, hi = np.nanpercentile(bs, [2.5, 97.5])
                pb = 2 * min(np.nanmean(bs <= 0), np.nanmean(bs >= 0))
                diff_rows.append({
                    "metric": "Overall (mean of 4 metrics)" if scope == "overall" else METRIC_LABEL[scope],
                    "coefficient": k, "comparison": name,
                    "judge_agreement": (ja + jb) / 2 if "mean(" in name else (ja if "Rad A  minus" in name else jb),
                    "inter_radiologist_agreement": ab,
                    "difference": est, "ci_low": lo, "ci_high": hi, "bootstrap_p_two_sided": min(1.0, pb),
                })
    diff_df = pd.DataFrame(diff_rows)
    diff_df.to_csv(os.path.join(OUT_DIR, "judge_vs_interrater_differences.csv"), index=False)

    # ---------------------------------------------------------------- consensus subset
    cons_rows = []
    for m in METRICS:
        q = N_CATEGORIES[m]
        W = weight_matrix(q, PRIMARY_WEIGHTS[m])
        a, b, g = labels["rad_a"][m], labels["rad_b"][m], labels["gpt_5"][m]
        S = np.flatnonzero(a == b)
        c_lab, g_lab = a[S], g[S]
        ns = S.size

        def stats_for(cl, gl):
            out = coefficients(gl, cl, q, W)
            classes = np.unique(cl)
            recalls = [np.mean(gl[cl == k] == k) for k in classes]
            out["accuracy"] = float(np.mean(gl == cl))
            out["balanced_accuracy"] = float(np.mean(recalls))
            out["majority_baseline"] = float(np.max(np.bincount(cl, minlength=q)) / cl.size)
            if m in PROBLEM_CLASS:
                pc = PROBLEM_CLASS[m]
                out["problem_detection_sensitivity"] = float(np.mean(gl[cl == pc] == pc)) if np.any(cl == pc) else np.nan
                out["problem_detection_specificity"] = float(np.mean(gl[cl != pc] != pc)) if np.any(cl != pc) else np.nan
            return out

        point = stats_for(c_lab, g_lab)
        keys = [k for k in point if k not in ("weighted_agreement",)]
        brng = np.random.default_rng([SEED, 7, METRICS.index(m)])
        boots = {k: [] for k in keys}
        for _ in range(B_BOOT):
            ii = brng.integers(0, ns, ns)
            sb = stats_for(c_lab[ii], g_lab[ii])
            for k in keys:
                boots[k].append(sb.get(k, np.nan))
        prng = np.random.default_rng([SEED, 8, METRICS.index(m)])
        wa_perm = np.array([coefficients(prng.permutation(g_lab), c_lab, q, W)["weighted_agreement"]
                            for _ in range(B_PERM)])
        rec = {
            "metric": METRIC_LABEL[m], "n_consensus_cases": int(ns), "n_total": n,
            "consensus_share": ns / n,
            "consensus_class_counts": {int(k): int(v) for k, v in zip(*np.unique(c_lab, return_counts=True))},
            "gpt5_class_counts_on_subset": {int(k): int(v) for k, v in zip(*np.unique(g_lab, return_counts=True))},
            "perm_p_beats_chance": float((1 + np.sum(wa_perm >= point["weighted_agreement"] - 1e-12)) / (B_PERM + 1)),
        }
        for k in keys:
            arr = np.asarray(boots[k], dtype=float)
            rec[k] = point[k]
            if np.isfinite(arr).any():
                rec[f"{k}_ci_low"], rec[f"{k}_ci_high"] = np.nanpercentile(arr, [2.5, 97.5])
        cons_rows.append(rec)
    cons_df = pd.DataFrame(cons_rows)
    cons_df.to_csv(os.path.join(OUT_DIR, "consensus_subset_validity.csv"), index=False)

    # ---------------------------------------------------------------- marginal bias
    bias_rows = []
    for m in METRICS:
        q = N_CATEGORIES[m]
        for p in PRIMARY_PAIRS:
            x, y = labels[p[0]][m], labels[p[1]][m]
            T = np.bincount(x * q + y, minlength=q * q).reshape(q, q)
            rec = {"metric": METRIC_LABEL[m], "pair": pair_label(p)}
            for r, v in [(p[0], x), (p[1], y)]:
                for k in range(q):
                    rec[f"{RATER_LABEL[r]}: P(score={k})"] = float(np.mean(v == k))
            if q == 2:
                res = mcnemar(T, exact=True)
                rec.update({"test": "McNemar (exact)", "statistic": float(res.statistic), "p_value": float(res.pvalue),
                            "bias_index": float((T[0, 1] - T[1, 0]) / T.sum()),
                            "prevalence_index": float((T[1, 1] - T[0, 0]) / T.sum())})
            else:
                res = SquareTable(T, shift_zeros=False).homogeneity()
                rec.update({"test": "Stuart-Maxwell", "statistic": float(res.statistic), "p_value": float(res.pvalue)})
            bias_rows.append(rec)
    bias_df = pd.DataFrame(bias_rows)
    bias_df.to_csv(os.path.join(OUT_DIR, "marginal_bias.csv"), index=False)

    # ---------------------------------------------------------------- console summary
    f = lambda v: f"{v:6.3f}"  # noqa: E731
    log("\n[AGREEMENT] primary pairs; [95% CI]; 'chance' = value expected under permutation null")
    for m in METRICS:
        log(f"\n  {METRIC_LABEL[m]}  ({PRIMARY_WEIGHTS[m]} weights)")
        sub = coef_df[(coef_df.metric == METRIC_LABEL[m]) & (coef_df.pair_type == "primary")]
        for _, r in sub.iterrows():
            log(f"    {r.pair:<32s} po={f(r.percent_agreement)}  kappa={f(r.kappa)} [{r.kappa_ci_low:5.2f},{r.kappa_ci_high:5.2f}]"
                f"  AC={f(r.gwet_ac)} [{r.gwet_ac_ci_low:5.2f},{r.gwet_ac_ci_high:5.2f}] chance={f(r.gwet_ac_chance_expected)}"
                f"  PABAK={f(r.brennan_prediger)}  F1={f(r.macro_f1)}  p(>chance)={r.perm_p_beats_chance:.4f}")
    log("\n[OVERALL] mean of 4 metrics")
    for _, r in overall_df[overall_df.pair_type == "primary"].iterrows():
        log(f"    {r.pair:<32s} kappa={f(r.kappa)} [{r.kappa_ci_low:5.2f},{r.kappa_ci_high:5.2f}]  "
            f"AC={f(r.gwet_ac)} [{r.gwet_ac_ci_low:5.2f},{r.gwet_ac_ci_high:5.2f}]  "
            f"PABAK={f(r.brennan_prediger)}  F1={f(r.macro_f1)} [{r.macro_f1_ci_low:5.2f},{r.macro_f1_ci_high:5.2f}]")
    log("\n[JUDGE - INTER-RADIOLOGIST] pooled difference, Gwet AC")
    for _, r in diff_df[(diff_df.coefficient == "gwet_ac") & diff_df.comparison.str.startswith("mean(")].iterrows():
        log(f"    {r.metric:<30s} judge={f(r.judge_agreement)} inter-rad={f(r.inter_radiologist_agreement)} "
            f"diff={r.difference:+.3f} [{r.ci_low:+.3f},{r.ci_high:+.3f}]")
    log("\n[CONSENSUS SUBSET] GPT-5 vs cases where the radiologists agree")
    for _, r in cons_df.iterrows():
        extra = ""
        if "problem_detection_sensitivity" in r and pd.notna(r.get("problem_detection_sensitivity")):
            extra = f"  problem sens={r.problem_detection_sensitivity:.2f} spec={r.problem_detection_specificity:.2f}"
        log(f"    {r.metric:<24s} n={r.n_consensus_cases:3d}  acc={r.accuracy:.3f} (majority {r.majority_baseline:.3f})  "
            f"bal.acc={r.balanced_accuracy:.3f} [{r.balanced_accuracy_ci_low:.2f},{r.balanced_accuracy_ci_high:.2f}]  "
            f"kappa={r.kappa:.3f}  p(>chance)={r.perm_p_beats_chance:.4f}{extra}")
    log("\n[MARGINAL BIAS]")
    for _, r in bias_df.iterrows():
        log(f"    {r.metric:<24s} {r.pair:<32s} {r.test:<16s} p={r.p_value:.3g}")

    log(f"\n[DONE] {time.time()-t0:.1f}s. Outputs in {OUT_DIR}")
    with open(LOG_PATH, "w") as fh:
        fh.write("\n".join(_log) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
