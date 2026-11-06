#!/usr/bin/env python3
r"""
Failure-mode analysis for term substitution (Figure 5 and the Failure Mode Analysis text).
No plotting.

A ground-truth substitution counts as REFLECTED ("detected") when the judge listed the
original term on the left-hand side of one of its detected alterations (exact, lower-cased
match). Miss rate = substitutions not reflected / substitutions evaluated.

Writes:
  failure_mode_records_<judge>.csv   one row per (score file, difficulty, summarizer, substituted term)
  category_miss_rates_<judge>.csv    per summarizer x category                         (Figure 5a)
  term_miss_rates_<judge>.csv        per summarizer x term                             (Figure 5b)

All detection and categorization functions are copied verbatim from the script used for the
paper (for names defined twice there, the definition Python actually uses is copied); only
the driver at the bottom is new. Note: summarizer keys parsed from GPT-5 score columns carry a
'_gpt' tail (e.g. 'claude_gpt'); the driver adds a cleaned 'summarizer' column.

Usage:
  python analysis/failure_mode_analysis.py --scores-dir data/judge_scores/gpt_5 --judge gpt_5 \
         --dictionary dictionary/adversarial_term_dictionary_v1_as_used.json --out-dir results/gpt_5
"""
import argparse
import ast
import glob
import json
import os
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ORIG_TO_CAT: Dict[str, str] = {}   # filled in main() from the dictionary, as in the original


# ======================================================================================
# Detection and categorization
# (verbatim from llm_as_judge_missed_terms_analysis.py)
# ======================================================================================
MEASUREMENT_TOKEN = re.compile(r'(\d+(?:\.\d+)?(?:\s*[x×]\s*\d+(?:\.\d+)?(?:\s*[x×]\s*\d+(?:\.\d+)?)?)?(?:\s*[–—-]\s*\d+(?:\.\d+)?)?\s*(?:mm|cm))', re.I)

GT_PREFIX  = "ground_truth_alter_terms_"

DET_PREFIX = "alterations_alter_terms_"

DET_COL_RE = re.compile(
    r"^alterations_alter_terms_(?P<diff>easy|medium|hard)_(?P<summ>.+?)_([^_]+)$",
    re.IGNORECASE
)

def parse_meta_from_path(path: str) -> Dict[str,str]:
    low = path.lower()
    meta = {
        "dataset":  "inspect" if "inspect" in low else "merlin" if "merlin" in low else "mimic" if "mimic" in low else "unknown",
        "judge":    "medgemma" if "medgemma" in low else "llama3_1" if "llama3_1" in low else "gpt_5" if "gpt" in low else "unknown",
        "prompt":   "long" if "/long/" in low or "_long_" in low else "short" if "/short/" in low or "_short_" in low else "unknown",
        "edit":     "alter_terms" if "alter_terms" in low else "missing_facts" if "missing" in low else "unknown",
        "difficulty":"hard" if "_hard_" in low else "medium" if "_medium_" in low else "easy" if "_easy_" in low else "unknown",
        "summarizer":"unknown",
    }
    # summarizer between "..._<difficulty>_" and "_summaries"
    m = re.search(r'_(easy|medium|hard)_([A-Za-z0-9_]+)_summaries', low)
    if m:
        meta["summarizer"] = m.group(2)
    return meta

def available_difficulties(df: pd.DataFrame) -> List[str]:
    return [d for d in ("easy","medium","hard") if f"{GT_PREFIX}{d}" in df.columns]

def extract_det_left_terms_from_cell(cell) -> List[str]:
    s = "" if cell is None else str(cell)
    # try literal-eval list first
    try:
        obj = ast.literal_eval(s)
    except Exception:
        obj = None
    out = []
    if isinstance(obj, (list, tuple)):
        for it in obj:
            if isinstance(it, str) and "->" in it:
                out.append(it.split("->", 1)[0].strip())
        if out: return out
    # free-text fallback: grab lhs of 'lhs -> rhs' patterns
    pairs = re.findall(r'([^,<>\n]+?)\s*->\s*[^,<>:\n]+', s)
    return [p.strip() for p in pairs if p.strip()]

def is_measurement_token(s: str) -> bool:
    return bool(MEASUREMENT_TOKEN.search(str(s)))

def get_category(term: str) -> str:
    t = term.strip().lower()
    if t in ORIG_TO_CAT:
        return ORIG_TO_CAT[t]
    # plural fallback
    # ts = _to_singular(t)
    ts = t
    if ts in ORIG_TO_CAT:
        return ORIG_TO_CAT[ts]
    if is_measurement_token(t):
        return 'measurement'
    for orig, cat in ORIG_TO_CAT.items():
        if orig in t or t in orig:
            return cat
    return 'unknown'

def collect_records_from_wide_fixed(df: pd.DataFrame,
                                    meta_base: Dict[str,str],
                                    log_unknown_terms: Optional[list] = None) -> pd.DataFrame:
    """
    For each available difficulty in this *wide* file:
      - GT = ground_truth_alter_terms_{diff}
      - DET = all columns matching alterations_alter_terms_{diff}_<summarizer>_<judgeTail>
    Detection rule: exact LHS (lowercased) equality (your original semantics).
    Terms whose category resolves to 'unknown' are skipped (optionally logged).
    """
    recs = []
    diffs = available_difficulties(df)
    if not diffs:
        return pd.DataFrame()

    for diff in diffs:
        gt_col = f"{GT_PREFIX}{diff}"
        gt_terms_series = df[gt_col].astype(str)

        # DET columns for this diff
        det_cols = []
        for c in df.columns:
            m = DET_COL_RE.match(str(c))
            if m and m.group("diff").lower() == diff:
                det_cols.append((c, m.group("summ").lower()))
        if not det_cols:
            continue

        for _, row in df.iterrows():
            # GT originals for THIS difficulty
            gt_terms = re.findall(r'\d+\.\s*(.+?)\s*->', str(row.get(gt_col, "")))
            gt_terms = [t.strip() for t in gt_terms if t.strip()]
            if not gt_terms:
                continue

            for det_col, summarizer in det_cols:
                det_left = extract_det_left_terms_from_cell(row.get(det_col, ""))
                det_lc   = {t.lower().strip() for t in det_left}

                for orig in gt_terms:
                    cat = get_category(orig)
                    if cat == "unknown":
                        if log_unknown_terms is not None:
                            log_unknown_terms.append({
                                **meta_base, "difficulty": diff, "summarizer": summarizer,
                                "term": orig.lower().strip(), "raw_orig": orig
                            })
                        continue  # skip unknown category terms entirely

                    o_lc = orig.lower().strip()
                    detected = (o_lc in det_lc)

                    recs.append({
                        **meta_base,
                        "difficulty": diff,
                        "summarizer": summarizer,
                        "category":   cat,
                        "term":       o_lc,
                        "detected":   1 if detected else 0,
                    })
    return pd.DataFrame(recs)

def build_records(files: List[str], unknown_terms_csv: Optional[str] = "unknown_terms_seen.csv") -> pd.DataFrame:
    all_parts = []
    unknown_log = [] if unknown_terms_csv else None

    for fp in files:
        try:
            df = pd.read_csv(fp)
        except Exception as e:
            print(f"[WARN] Failed to read {fp}: {e}")
            continue

        meta = parse_meta_from_path(fp)
        # build recs from this wide file
        rec = collect_records_from_wide_fixed(df, meta_base=meta, log_unknown_terms=unknown_log)
        if rec.empty:
            print(f"[DEBUG] No records in {os.path.basename(fp)}")
            continue
        all_parts.append(rec)

    out = pd.concat(all_parts, ignore_index=True) if all_parts else pd.DataFrame()
    if unknown_log is not None and unknown_log:
        pd.DataFrame(unknown_log).to_csv(unknown_terms_csv, index=False)
        print(f"[INFO] Wrote unknown terms log: {unknown_terms_csv} ({len(unknown_log)} rows)")
    return out

# ==========================================================================================
# Driver (new; no plotting)
# ==========================================================================================
def main() -> None:
    global ORIG_TO_CAT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores-dir", required=True,
                    help="Folder with <judge>_<dataset>_<long|short>_alter_terms_combined_cleaned_wide.csv")
    ap.add_argument("--judge", default="gpt_5", help="Judge prefix of the score files (gpt_5 or medgemma)")
    ap.add_argument("--dictionary", default="dictionary/adversarial_term_dictionary_v1_as_used.json",
                    help="Term dictionary used to generate the substitutions (v1 for the paper)")
    ap.add_argument("--out-dir", required=True, help="Output folder for the tables")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.dictionary, "r") as f:
        mapping_raw = json.load(f).get("term_alternatives", {})
    ORIG_TO_CAT = {k.lower(): v.get("category", "unknown") for k, v in mapping_raw.items()}

    files = sorted(glob.glob(os.path.join(args.scores_dir, f"{args.judge}_*_alter_terms_combined_cleaned_wide.csv")))
    if not files:
        raise SystemExit(f"No {args.judge}_*_alter_terms_combined_cleaned_wide.csv files in {args.scores_dir}")
    rec = build_records(files, unknown_terms_csv=None)
    rec["summarizer_key"] = rec["summarizer"]
    rec["summarizer"] = rec["summarizer_key"].str.replace(r"_gpt$", "", regex=True)
    rec.to_csv(os.path.join(args.out_dir, f"failure_mode_records_{args.judge}.csv"), index=False)

    def rate_table(group_cols):
        t = rec.groupby(group_cols, as_index=False).agg(total=("detected", "count"), reflected=("detected", "sum"))
        t["missed"] = t["total"] - t["reflected"]
        t["miss_rate"] = t["missed"] / t["total"]
        return t

    cat = rate_table(["summarizer", "category"]).sort_values(["summarizer", "miss_rate"], ascending=[True, False])
    cat.to_csv(os.path.join(args.out_dir, f"category_miss_rates_{args.judge}.csv"), index=False)
    term = rate_table(["summarizer", "term", "category"]).sort_values(
        ["summarizer", "missed", "miss_rate"], ascending=[True, False, False])
    term.to_csv(os.path.join(args.out_dir, f"term_miss_rates_{args.judge}.csv"), index=False)
    print(f"[OK] {len(files)} score files, {len(rec):,} substitution records -> {args.out_dir}")
    print("     Figure 5a used categories with >= 3 substitutions; Figure 5b terms with >= 5.")


if __name__ == "__main__":
    main()
