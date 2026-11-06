#!/usr/bin/env python3
r"""
Confirmatory multi-factor modelling of the GPT-5 judge scores.

Aims (referenced as A1-A4 in the code below):
  A1  exploit the factorial structure with a multi-factor model rather than
      separate pairwise permutation tests
  A2  formally examine interaction effects (prompt x model, prompt x edit type,
      prompt x dataset, model x edit type, model x dataset, ...)
  A3  use model-based estimation for binary and ordinal outcomes, giving
      efficient estimates, confidence intervals and formal interaction tests
  A4  report an omnibus analysis before any pairwise comparison, so that the
      pairwise contrasts sit inside a single inferential framework

DESIGN (verified by build_model_dataset.py)
    Balanced repeated-measures factorial. 300 reports (100 per dataset), each
    evaluated under every combination of
        dataset(3) x prompt(2) x summarizer(6) x condition(5)
    condition = edit type x intensity:
        AlterTerm-Easy/-Medium/-Hard, MissingFact-Easy/-Hard
    18,000 observations per metric; 60 per report. Repeated measures on the report
    are handled throughout (GEE working correlation + report-clustered robust
    variance; random-intercept GLMM as a sensitivity fit).

MODELS
    Binary outcomes (factual faithfulness, certainty fidelity, hallucination)
        population-averaged logistic regression fitted by GEE, exchangeable
        working correlation, clusters = report, robust (sandwich) covariance.
    Ordinal outcome (sensitivity to changes, 0/1/2)
        proportional-odds cumulative logit, report-clustered sandwich covariance.

    Nested hierarchy, fitted for every outcome:
        M1  main effects                 summarizer + prompt + condition + dataset
        M2  + all two-way interactions   (the pre-specified omnibus model)
        M3  + summarizer:prompt:condition
    Factors are sum-to-zero coded in the omnibus fits so that the Wald tests of
    lower-order terms are genuine Type III tests in the presence of interactions.
    A parallel treatment-coded fit of M2 supplies interpretable odds ratios.

OUTPUT (all written to --out-dir)
    omnibus_type3_wald.csv          Type III Wald test per model term, per outcome
    edit_type_intensity_decomposition.csv
                                    edit type and edit intensity recovered as explicit
                                    contrasts of the condition factor
    model_hierarchy_tests.csv       joint tests of interaction blocks + QIC
    coefficients_or.csv             treatment-coded odds ratios with 95% CI
    emmeans_*.csv                   model-based marginal estimates with 95% CI
    contrasts_prompt_by_summarizer.csv   prompt effect within summarizer x edit type
    contrasts_summarizer_pairwise.csv    pairwise summarizer contrasts, Holm adjusted
    glmm_random_intercept.csv       random-intercept GLMM variance component + ICC
    proportional_odds_check.csv     partial-proportional-odds diagnostic
    model_fit_log.txt               convergence / runtime log

Usage:
    python analysis/multifactor_models/fit_models.py \
        --data results/multifactor_models/model_dataset_gpt5_judge.csv --out-dir results/multifactor_models
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.cov_struct import Exchangeable, Independence
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.stats.multitest import multipletests

# Set from the command line in main().
OUT_DIR = "results/multifactor_models"
DATA = os.path.join(OUT_DIR, "model_dataset_gpt5_judge.csv")
LOG_PATH = os.path.join(OUT_DIR, "model_fit_log.txt")

BINARY_OUTCOMES = ["Factual Faithfulness", "Certainty Fidelity", "Hallucination (present)"]
ORDINAL_OUTCOME = "Sensitivity to Changes"
ALL_OUTCOMES = BINARY_OUTCOMES + [ORDINAL_OUTCOME]

# Reference levels for the treatment-coded (interpretable) fit.
REF = {
    "summarizer_label": "GPT-5-mini",
    "prompt_label": "Concise",       # so the prompt effect reads Detailed - Concise
    "condition": "AlterTerm-Easy",
    "dataset_label": "MERLIN",
}

FACTORS = ["summarizer_label", "prompt_label", "condition", "dataset_label"]
PRETTY_FACTOR = {
    "summarizer_label": "summarizer",
    "prompt_label": "prompt",
    "condition": "condition",
    "dataset_label": "dataset",
}

_log_lines: list[str] = []


def log(msg: str = "") -> None:
    print(msg)
    _log_lines.append(msg)


# --------------------------------------------------------------------------------------
# formulas
# --------------------------------------------------------------------------------------
def _terms(coding: str) -> dict[str, str]:
    """Factor expressions under sum-to-zero ('sum') or treatment ('treatment') coding."""
    out = {}
    for f in FACTORS:
        if coding == "sum":
            out[f] = f"C({f}, Sum)"
        else:
            out[f] = f"C({f}, Treatment(reference='{REF[f]}'))"
    return out


def formulas(coding: str) -> dict[str, str]:
    t = _terms(coding)
    s, p, c, d = t["summarizer_label"], t["prompt_label"], t["condition"], t["dataset_label"]
    main = f"{s} + {p} + {c} + {d}"
    two_way = " + ".join(f"{a}:{b}" for a, b in combinations([s, p, c, d], 2))
    three = f"{s}:{p}:{c}"
    return {
        "M1_main_effects": main,
        "M2_two_way": f"{main} + {two_way}",
        "M3_three_way": f"{main} + {two_way} + {three}",
    }


def term_label(term_name: str) -> str:
    """Turn a patsy term name back into readable factor names."""
    out = term_name
    for f in FACTORS:
        out = out.replace(f"C({f}, Sum)", PRETTY_FACTOR[f])
        out = out.replace(f"C({f}, Treatment(reference='{REF[f]}'))", PRETTY_FACTOR[f])
    return out.replace(":", " x ")


# --------------------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------------------
def fit_binary_gee(df: pd.DataFrame, rhs: str, cov_struct=None):
    """Population-averaged logistic GEE, clusters = report, robust covariance."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.gee(
            formula=f"outcome ~ {rhs}",
            groups="report_id",
            data=df,
            family=Binomial(),
            cov_struct=Exchangeable() if cov_struct is None else cov_struct,
        )
        res = model.fit(maxiter=120)
    return res


def _design_no_intercept(rhs: str, df: pd.DataFrame):
    """Patsy design with the intercept column dropped but reduced-rank coding kept."""
    X = patsy.dmatrix(f"1 + {rhs}", df, return_type="dataframe")
    di = X.design_info
    X = X.drop(columns=["Intercept"])
    return X, di


def fit_ordinal(df: pd.DataFrame, rhs: str):
    """Proportional-odds cumulative logit with report-clustered sandwich covariance."""
    X, di = _design_no_intercept(rhs, df)
    y = df["outcome"].astype(int).values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = OrderedModel(y, X.values, distr="logit")
        start = mod.fit(method="bfgs", disp=0, maxiter=400)
        res = mod.fit(
            method="bfgs",
            disp=0,
            maxiter=400,
            start_params=start.params,
            cov_type="cluster",
            cov_kwds={"groups": df["report_id"].values},
        )
    res._design_info = di
    res._exog_names = list(X.columns)
    return res


# --------------------------------------------------------------------------------------
# Type III Wald tests
# --------------------------------------------------------------------------------------
def _term_slices(res, ordinal: bool):
    """Map each model term -> the parameter indices it occupies."""
    if ordinal:
        di = res._design_info
        names = res._exog_names
        idx = {n: i for i, n in enumerate(names)}
        out = {}
        for term, sl in di.term_name_slices.items():
            if term == "Intercept":
                continue
            cols = [c for c in di.column_names[sl] if c in idx]
            if cols:
                out[term] = [idx[c] for c in cols]
        return out
    di = res.model.data.design_info
    out = {}
    for term, sl in di.term_name_slices.items():
        if term == "Intercept":
            continue
        out[term] = list(range(sl.start, sl.stop))
    return out


def type3_wald(res, ordinal: bool = False) -> pd.DataFrame:
    """Joint Wald chi-square test for every model term (sum coding -> Type III)."""
    slices = _term_slices(res, ordinal)
    nparam = len(res.params)
    rows = []
    for term, idxs in slices.items():
        R = np.zeros((len(idxs), nparam))
        for r, i in enumerate(idxs):
            R[r, i] = 1.0
        wt = res.wald_test(R, scalar=True, use_f=False)
        rows.append(
            {
                "term": term_label(term),
                "df": int(len(idxs)),
                "wald_chi2": float(np.squeeze(wt.statistic)),
                "p_value": float(np.squeeze(wt.pvalue)),
            }
        )
    return pd.DataFrame(rows)


def joint_wald(res, term_filter, ordinal: bool = False):
    """Joint Wald test over every term whose label satisfies term_filter."""
    slices = _term_slices(res, ordinal)
    idxs = []
    for term, ii in slices.items():
        if term_filter(term_label(term)):
            idxs.extend(ii)
    if not idxs:
        return None
    R = np.zeros((len(idxs), len(res.params)))
    for r, i in enumerate(sorted(idxs)):
        R[r, i] = 1.0
    wt = res.wald_test(R, scalar=True, use_f=False)
    return int(len(idxs)), float(np.squeeze(wt.statistic)), float(np.squeeze(wt.pvalue))


# --------------------------------------------------------------------------------------
# estimated marginal means / contrasts on the link scale
# --------------------------------------------------------------------------------------
def design_grid(df: pd.DataFrame) -> pd.DataFrame:
    """The 180 balanced cells of the factorial design."""
    levels = {f: sorted(df[f].unique()) for f in FACTORS}
    grid = pd.MultiIndex.from_product([levels[f] for f in FACTORS], names=FACTORS).to_frame(index=False)
    assert len(grid) == 180, len(grid)
    return grid


def grid_exog(res, grid: pd.DataFrame, ordinal: bool) -> np.ndarray:
    """Model matrix rows for the design grid (linear-predictor part only)."""
    di = res._design_info if ordinal else res.model.data.design_info
    X = patsy.build_design_matrices([di], grid, return_type="dataframe")[0]
    if ordinal:
        X = X.drop(columns=["Intercept"])
    return X.values


def grid_matrix(res, grid: pd.DataFrame, ordinal: bool) -> np.ndarray:
    """Design-grid rows padded to the full parameter vector, for linear contrasts."""
    X = grid_exog(res, grid, ordinal)
    if ordinal:
        M = np.zeros((len(grid), len(res.params)))
        M[:, : X.shape[1]] = X              # thresholds occupy the trailing params
        return M
    return X


def avg_response(res, Xg: np.ndarray, ordinal: bool):
    """
    Marginally standardised prediction: the average fitted response over a set of
    design cells, with a delta-method standard error. For the binary models this is
    an average predicted probability; for the ordinal model it is the expected score
    on the original 0-2 scale. Unlike the link-scale EMM it is directly comparable
    with the descriptive rates reported in the manuscript.
    """
    def g(p: np.ndarray) -> float:
        if ordinal:
            pr = np.asarray(res.model.predict(p, exog=Xg))
            return float(np.mean(pr @ np.arange(pr.shape[1])))
        eta = Xg @ p
        return float(np.mean(1.0 / (1.0 + np.exp(-eta))))

    b = np.asarray(res.params, dtype=float)
    est = g(b)
    step = 1e-6 * np.maximum(1.0, np.abs(b))
    grad = np.empty_like(b)
    for i in range(b.size):
        bp, bm = b.copy(), b.copy()
        bp[i] += step[i]
        bm[i] -= step[i]
        grad[i] = (g(bp) - g(bm)) / (2 * step[i])
    V = np.asarray(res.cov_params(), dtype=float)
    se = float(np.sqrt(max(grad @ V @ grad, 0.0)))
    return est, se


def _lincom(res, c: np.ndarray):
    """Estimate, SE, z, p and 95% CI for the linear combination c'beta."""
    b = np.asarray(res.params, dtype=float)
    V = np.asarray(res.cov_params(), dtype=float)
    est = float(c @ b)
    se = float(np.sqrt(c @ V @ c))
    z = est / se if se > 0 else np.nan
    p = float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan
    return est, se, z, p, est - 1.959963985 * se, est + 1.959963985 * se


def emmeans(res, grid: pd.DataFrame, by: list[str], ordinal: bool,
            observed: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Model-based marginal estimates, equally weighting the cells of the balanced
    design. Two complementary summaries are reported for each cell group:

      estimate_link / *_link   the link-scale EMM (emmeans convention) and, for the
                               binary models, its back-transform. Contrasts and
                               p-values are computed on this scale.
      estimate_response        the marginally standardised average fitted response,
                               which is on the same footing as the descriptive rates
                               in the manuscript (probability, or expected 0-2 score).
    """
    M = grid_matrix(res, grid, ordinal)
    Xg = grid_exog(res, grid, ordinal)
    rows = []
    for key, sub in grid.groupby(by, sort=True):
        idx = sub.index.values
        c = M[idx].mean(axis=0)
        est, se, z, p, lo, hi = _lincom(res, c)
        rec = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        rec.update({"n_cells": len(idx), "estimate_link": est, "se_link": se,
                    "ci_low_link": lo, "ci_high_link": hi})
        if not ordinal:
            rec.update({
                "estimate_prob_emm": float(1 / (1 + np.exp(-est))),
                "ci_low_prob_emm": float(1 / (1 + np.exp(-lo))),
                "ci_high_prob_emm": float(1 / (1 + np.exp(-hi))),
            })
        r_est, r_se = avg_response(res, Xg[idx], ordinal)
        rec.update({
            "estimate_response": r_est,
            "se_response": r_se,
            "ci_low_response": r_est - 1.959963985 * r_se,
            "ci_high_response": r_est + 1.959963985 * r_se,
        })
        if observed is not None:
            m = np.ones(len(observed), dtype=bool)
            for col, val in zip(by, key if isinstance(key, tuple) else (key,)):
                m &= observed[col].values == val
            rec["observed_mean"] = float(observed.loc[m, "outcome"].mean())
        rows.append(rec)
    return pd.DataFrame(rows)


def contrast(res, grid: pd.DataFrame, mask_a: np.ndarray, mask_b: np.ndarray, ordinal: bool):
    """Link-scale difference between two equally weighted sets of design cells."""
    M = grid_matrix(res, grid, ordinal)
    c = M[mask_a].mean(axis=0) - M[mask_b].mean(axis=0)
    return _lincom(res, c)


def joint_contrast_test(res, rows: list[np.ndarray]):
    """Joint Wald chi-square test that every supplied contrast is zero."""
    R = np.vstack(rows)
    wt = res.wald_test(R, scalar=True, use_f=False)
    return int(R.shape[0]), float(np.squeeze(wt.statistic)), float(np.squeeze(wt.pvalue))


def decompose_condition(res, grid: pd.DataFrame, ordinal: bool) -> pd.DataFrame:
    """
    The design has five named factors: dataset, summarizer, prompt type,
    adversarial-edit TYPE and edit INTENSITY. The omnibus model carries edit type and
    intensity inside a single five-level `condition` factor, because impression removal
    was generated at only two intensities and a fully crossed type x intensity factor
    would therefore contain an empty cell.

    This function recovers the named factors as explicit linear contrasts of that same
    fitted model, so they are tested without refitting anything and without unbalancing
    the design:
        edit type                 term substitution vs impression removal (1 df)
        intensity | term subst.   easy vs medium vs hard                  (2 df)
        intensity | impression    easy vs hard                            (1 df)
        edit type x prompt / x summarizer / x dataset
    """
    M = grid_matrix(res, grid, ordinal)
    cond = grid["condition"].values
    edit = np.array([c.split("-")[0] for c in cond])
    inten = np.array([c.split("-")[1] for c in cond])
    summ = grid["summarizer_label"].values
    prom = grid["prompt_label"].values
    dset = grid["dataset_label"].values

    def cv(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return M[a].mean(axis=0) - M[b].mean(axis=0)

    alter, missing = edit == "AlterTerm", edit == "MissingFact"
    rows = []

    def add(name, contrast_rows):
        df_, chi2, p = joint_contrast_test(res, contrast_rows)
        rows.append({"term": name, "df": df_, "wald_chi2": chi2, "p_value": p})

    add("edit type (substitution vs removal)", [cv(alter, missing)])
    add("intensity | term substitution", [
        cv(alter & (inten == "Medium"), alter & (inten == "Easy")),
        cv(alter & (inten == "Hard"), alter & (inten == "Medium")),
    ])
    add("intensity | impression removal", [cv(missing & (inten == "Hard"), missing & (inten == "Easy"))])

    # edit type x prompt: does the edit-type effect differ between prompts?
    add("edit type x prompt", [
        cv(alter & (prom == "Detailed"), missing & (prom == "Detailed"))
        - cv(alter & (prom == "Concise"), missing & (prom == "Concise"))
    ])

    # edit type x summarizer / x dataset: edit-type effect for each level vs the last
    for label, arr in [("summarizer", summ), ("dataset", dset)]:
        levels = sorted(set(arr))
        ref = levels[-1]
        ref_eff = cv(alter & (arr == ref), missing & (arr == ref))
        add(f"edit type x {label}",
            [cv(alter & (arr == lv), missing & (arr == lv)) - ref_eff for lv in levels[:-1]])

    return pd.DataFrame(rows)


def proportional_odds_check(df: pd.DataFrame, rhs: str, ordinal_t3: pd.DataFrame) -> pd.DataFrame:
    """
    Diagnostic for the proportional-odds assumption of the ordinal model.

    The cumulative logit model assumes one common set of coefficients for both binary
    splits of the 0/1/2 score. We refit the same model separately as P(Y>=1) and
    P(Y>=2) logistic GEEs and compare. This is a parallel-lines diagnostic, NOT a formal
    Brant test: the two split models are fitted to the same reports, so their joint
    covariance is not available here and no omnibus statistic is computed. What it does
    establish is whether the substantive conclusions - which terms matter - depend on
    the assumption.
    """
    out = {}
    coefs = {}
    names = None
    for thr, lab in [(1, "ge1"), (2, "ge2")]:
        d = df.copy()
        d["outcome"] = (d["outcome"].values >= thr).astype(float)
        res = fit_binary_gee(d, rhs)
        t = type3_wald(res).set_index("term")
        out[lab] = t
        coefs[lab] = np.asarray(res.params, dtype=float)
        names = list(res.model.exog_names)

    o = ordinal_t3.set_index("term")
    rows = []
    for term in o.index:
        r = {
            "term": term,
            "df": int(o.loc[term, "df"]),
            "ordinal_chi2": float(o.loc[term, "wald_chi2"]),
            "ordinal_p": float(o.loc[term, "p_value"]),
        }
        for lab in ("ge1", "ge2"):
            r[f"{lab}_chi2"] = float(out[lab].loc[term, "wald_chi2"])
            r[f"{lab}_p"] = float(out[lab].loc[term, "p_value"])
        sig = [r["ordinal_p"] < 0.05, r["ge1_p"] < 0.05, r["ge2_p"] < 0.05]
        r["significance_agrees_across_splits"] = bool(len(set(sig)) == 1)
        rows.append(r)

    res_df = pd.DataFrame(rows)
    res_df["coef_corr_ge1_vs_ge2"] = float(np.corrcoef(coefs["ge1"], coefs["ge2"])[0, 1])
    keep = [j for j, nm in enumerate(names) if nm != "Intercept"]
    res_df["coef_corr_ge1_vs_ge2_no_intercept"] = float(
        np.corrcoef(coefs["ge1"][keep], coefs["ge2"][keep])[0, 1]
    )
    return res_df


# --------------------------------------------------------------------------------------
# random-intercept GLMM sensitivity fit
# --------------------------------------------------------------------------------------
def fit_glmm(df: pd.DataFrame, rhs: str):
    """
    Random-intercept logistic GLMM (report-level intercept), fitted by variational
    Bayes. Used only to confirm the fixed-effect conclusions and to report the
    report-level variance component / ICC.
    """
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = BinomialBayesMixedGLM.from_formula(
            f"outcome ~ {rhs}", {"report": "0 + C(report_id)"}, df
        )
        res = model.fit_vb(verbose=False)
    return res


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def main() -> int:
    global OUT_DIR, DATA, LOG_PATH
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DATA, help="model_dataset_gpt5_judge.csv from build_model_dataset.py")
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()
    DATA, OUT_DIR = args.data, args.out_dir
    LOG_PATH = os.path.join(OUT_DIR, "model_fit_log.txt")
    os.makedirs(OUT_DIR, exist_ok=True)
    t_all = time.time()
    df_all = pd.read_csv(DATA)
    log(f"[DATA] {DATA}")
    log(f"[DATA] {len(df_all):,} rows | judge = {df_all['judge'].unique().tolist()}")
    log("")

    grid = None
    omnibus_rows, hierarchy_rows, coef_rows = [], [], []
    emm_main, emm_prompt_summ, emm_cond_summ = [], [], []
    prompt_contrasts, summ_contrasts = [], []
    glmm_rows, decomposition_rows, po_rows = [], [], []

    for outcome in ALL_OUTCOMES:
        ordinal = outcome == ORDINAL_OUTCOME
        df = df_all[df_all["outcome_name"] == outcome].copy()
        df = df.sort_values("report_id").reset_index(drop=True)
        if grid is None:
            grid = design_grid(df)
        log("=" * 88)
        log(f"OUTCOME: {outcome}   ({'ordinal 0/1/2 proportional-odds' if ordinal else 'binary logistic GEE'})")
        log(f"  n = {len(df):,}   reports = {df['report_id'].nunique()}   "
            f"mean outcome = {df['outcome'].mean():.4f}")

        fs_sum = formulas("sum")
        fitted = {}
        for name, rhs in fs_sum.items():
            t0 = time.time()
            res = fit_ordinal(df, rhs) if ordinal else fit_binary_gee(df, rhs)
            fitted[name] = res
            npar = len(res.params)
            log(f"  [{name}] fitted in {time.time()-t0:6.1f}s, {npar} parameters")

        # ---- A4 omnibus: Type III Wald tests from the pre-specified M2 -------------
        res2 = fitted["M2_two_way"]
        t3 = type3_wald(res2, ordinal=ordinal)
        t3.insert(0, "outcome", outcome)
        t3.insert(1, "model", "M2_two_way")
        omnibus_rows.append(t3)
        log("\n  Type III Wald tests (M2: main effects + all two-way interactions)")
        for _, r in t3.iterrows():
            log(f"    {r['term']:<34s} df={r['df']:>3d}  chi2={r['wald_chi2']:10.2f}  p={r['p_value']:.3e}")

        # ---- A1 named factors: edit type and edit intensity, as contrasts ----------
        dec = decompose_condition(res2, grid, ordinal)
        dec.insert(0, "outcome", outcome)
        dec.insert(1, "model", "M2_two_way (contrasts of the condition factor)")
        decomposition_rows.append(dec)
        log("\n  Edit type / edit intensity decomposition of the condition factor")
        for _, r in dec.iterrows():
            log(f"    {r['term']:<40s} df={r['df']:>3d}  chi2={r['wald_chi2']:9.2f}  p={r['p_value']:.3e}")

        # ---- proportional-odds diagnostic for the ordinal model ----------------------
        if ordinal:
            po = proportional_odds_check(df, fs_sum["M2_two_way"], t3)
            po.insert(0, "outcome", outcome)
            po_rows.append(po)
            agree = int(po["significance_agrees_across_splits"].sum())
            log(f"\n  Proportional-odds diagnostic: significance conclusions agree for "
                f"{agree}/{len(po)} terms across the P(Y>=1) and P(Y>=2) splits; "
                f"slope correlation r = {po['coef_corr_ge1_vs_ge2_no_intercept'].iloc[0]:.3f} "
                f"(r = {po['coef_corr_ge1_vs_ge2'].iloc[0]:.3f} with intercept). "
                f"A moderate correlation indicates the proportional-odds assumption is "
                f"imperfect: the single OR is an average effect across thresholds.")

        # ---- A2/A4 joint tests of the interaction blocks -------------------------
        jt = joint_wald(res2, lambda t: " x " in t, ordinal=ordinal)
        if jt:
            hierarchy_rows.append({
                "outcome": outcome, "model": "M2_two_way",
                "test": "all two-way interactions jointly zero",
                "df": jt[0], "wald_chi2": jt[1], "p_value": jt[2],
            })
        res3 = fitted["M3_three_way"]
        jt3 = joint_wald(res3, lambda t: t.count(" x ") == 2, ordinal=ordinal)
        if jt3:
            hierarchy_rows.append({
                "outcome": outcome, "model": "M3_three_way",
                "test": "summarizer x prompt x condition jointly zero",
                "df": jt3[0], "wald_chi2": jt3[1], "p_value": jt3[2],
            })
        if not ordinal:
            # Pan (2001): QIC must be computed with a COMMON scale to be comparable
            # across nested models; use the scale estimated under the largest model.
            scale_ref = float(fitted["M3_three_way"].scale)
            for name in ["M1_main_effects", "M2_two_way", "M3_three_way"]:
                q = fitted[name].qic(scale=scale_ref)
                hierarchy_rows.append({
                    "outcome": outcome, "model": name,
                    "test": f"QIC (lower = better; common scale={scale_ref:.4f})",
                    "df": len(fitted[name].params), "wald_chi2": np.nan, "p_value": np.nan,
                    "qic": float(q[0]), "qicu": float(q[1]),
                })
        else:
            for name in ["M1_main_effects", "M2_two_way", "M3_three_way"]:
                r = fitted[name]
                hierarchy_rows.append({
                    "outcome": outcome, "model": name, "test": "naive log-likelihood / AIC",
                    "df": len(r.params), "wald_chi2": np.nan, "p_value": np.nan,
                    "loglike": float(r.llf), "aic": float(r.aic),
                })

        # ---- A3 interpretable effect sizes: treatment-coded M2 ---------------------
        rhs_t = formulas("treatment")["M2_two_way"]
        res_t = fit_ordinal(df, rhs_t) if ordinal else fit_binary_gee(df, rhs_t)
        if ordinal:
            n_thresh = len(res_t.params) - len(res_t._exog_names)
            names = list(res_t._exog_names) + [f"threshold_{i}" for i in range(n_thresh)]
        else:
            names = list(res_t.model.exog_names)
        b = np.asarray(res_t.params, dtype=float)
        se = np.sqrt(np.diag(np.asarray(res_t.cov_params(), dtype=float)))
        z = np.divide(b, se, out=np.full_like(b, np.nan), where=se > 0)
        coef_rows.append(pd.DataFrame({
            "outcome": outcome,
            "model": "M2_two_way (treatment coding)",
            "term": [term_label(n) for n in names],
            "coef_logit": b,
            "se": se,
            "z": z,
            "p_value": 2 * stats.norm.sf(np.abs(z)),
            "odds_ratio": np.exp(b),
            "or_ci_low": np.exp(b - 1.959963985 * se),
            "or_ci_high": np.exp(b + 1.959963985 * se),
        }))

        # ---- model-based marginal estimates -----------------------------------------
        e1 = emmeans(res2, grid, ["summarizer_label"], ordinal, observed=df)
        e1.insert(0, "outcome", outcome)
        emm_main.append(e1)

        e2 = emmeans(res2, grid, ["summarizer_label", "prompt_label"], ordinal, observed=df)
        e2.insert(0, "outcome", outcome)
        emm_prompt_summ.append(e2)

        e3 = emmeans(res2, grid, ["summarizer_label", "condition"], ordinal, observed=df)
        e3.insert(0, "outcome", outcome)
        emm_cond_summ.append(e3)

        # ---- A2 prompt effect within summarizer x edit type ------------------------
        # (the model-based counterpart of the published permutation tests)
        grid_edit = grid["condition"].str.split("-").str[0].map(
            {"AlterTerm": "Alter Term", "MissingFact": "Missing Fact"}
        ).values
        rows = []
        for summ in sorted(grid["summarizer_label"].unique()):
            for edit in ["Alter Term", "Missing Fact"]:
                base = (grid["summarizer_label"].values == summ) & (grid_edit == edit)
                a = base & (grid["prompt_label"].values == "Detailed")
                bb = base & (grid["prompt_label"].values == "Concise")
                est, s, zz, p, lo, hi = contrast(res2, grid, a, bb, ordinal)
                rows.append({
                    "outcome": outcome, "summarizer_label": summ, "edit_type": edit,
                    "contrast": "Detailed - Concise",
                    "log_or": est, "se": s, "z": zz, "p_value": p,
                    "odds_ratio": float(np.exp(est)),
                    "or_ci_low": float(np.exp(lo)), "or_ci_high": float(np.exp(hi)),
                })
        pc = pd.DataFrame(rows)
        pc["p_holm_within_outcome"] = multipletests(pc["p_value"], method="holm")[1]
        prompt_contrasts.append(pc)

        # ---- pairwise summarizer contrasts (only after the omnibus) ------------------
        sig = float(t3.loc[t3["term"] == "summarizer", "p_value"].iloc[0])
        rows = []
        summs = sorted(grid["summarizer_label"].unique())
        for a, bl in combinations(summs, 2):
            ma = grid["summarizer_label"].values == a
            mb = grid["summarizer_label"].values == bl
            est, s, zz, p, lo, hi = contrast(res2, grid, ma, mb, ordinal)
            rows.append({
                "outcome": outcome, "omnibus_summarizer_p": sig,
                "contrast": f"{a} - {bl}",
                "log_or": est, "se": s, "z": zz, "p_value": p,
                "odds_ratio": float(np.exp(est)),
                "or_ci_low": float(np.exp(lo)), "or_ci_high": float(np.exp(hi)),
            })
        sc = pd.DataFrame(rows)
        sc["p_holm_within_outcome"] = multipletests(sc["p_value"], method="holm")[1]
        summ_contrasts.append(sc)

        # ---- random-intercept GLMM sensitivity fit -----------------------------------
        if not ordinal:
            t0 = time.time()
            try:
                gres = fit_glmm(df, fs_sum["M2_two_way"])
                sd = float(np.exp(gres.vcp_mean[0]))
                icc = sd ** 2 / (sd ** 2 + np.pi ** 2 / 3)
                # correlation between GEE and GLMM fixed effects
                k = min(len(gres.fe_mean), len(res2.params))
                corr = float(np.corrcoef(np.asarray(gres.fe_mean)[:k], np.asarray(res2.params)[:k])[0, 1])
                glmm_rows.append({
                    "outcome": outcome,
                    "model": "random intercept (1|report), M2 fixed effects, variational Bayes",
                    "report_sd_logit": sd,
                    "report_variance_logit": sd ** 2,
                    "icc_latent_scale": icc,
                    "n_reports": int(df["report_id"].nunique()),
                    "corr_fixed_effects_with_gee": corr,
                    "fit_seconds": time.time() - t0,
                })
                log(f"\n  GLMM random intercept: SD={sd:.3f}, ICC={icc:.3f}, "
                    f"fixed-effect corr with GEE = {corr:.4f}  ({time.time()-t0:.1f}s)")
            except Exception as exc:  # pragma: no cover
                log(f"  [WARN] GLMM fit failed: {type(exc).__name__}: {exc}")
                glmm_rows.append({"outcome": outcome, "model": "FAILED", "error": str(exc)})
        log("")

    # ---- write everything ------------------------------------------------------------
    def write(frames, name):
        out = pd.concat(frames, ignore_index=True)
        p = os.path.join(OUT_DIR, name)
        out.to_csv(p, index=False)
        log(f"[OK] {name}  ({len(out)} rows)")
        return out

    write(omnibus_rows, "omnibus_type3_wald.csv")
    write(decomposition_rows, "edit_type_intensity_decomposition.csv")
    if po_rows:
        write(po_rows, "proportional_odds_check.csv")
    pd.DataFrame(hierarchy_rows).to_csv(os.path.join(OUT_DIR, "model_hierarchy_tests.csv"), index=False)
    log(f"[OK] model_hierarchy_tests.csv  ({len(hierarchy_rows)} rows)")
    write(coef_rows, "coefficients_or.csv")
    write(emm_main, "emmeans_summarizer.csv")
    write(emm_prompt_summ, "emmeans_summarizer_by_prompt.csv")
    write(emm_cond_summ, "emmeans_summarizer_by_condition.csv")
    write(prompt_contrasts, "contrasts_prompt_by_summarizer.csv")
    write(summ_contrasts, "contrasts_summarizer_pairwise.csv")
    if glmm_rows:
        pd.DataFrame(glmm_rows).to_csv(os.path.join(OUT_DIR, "glmm_random_intercept.csv"), index=False)
        log(f"[OK] glmm_random_intercept.csv  ({len(glmm_rows)} rows)")

    log(f"\n[DONE] total {time.time()-t_all:.1f}s")
    with open(LOG_PATH, "w") as fh:
        fh.write("\n".join(_log_lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
