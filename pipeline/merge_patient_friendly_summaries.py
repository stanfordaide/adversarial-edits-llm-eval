#!/usr/bin/env python3
r"""
Step 3 of the pipeline: merge the original and adversarial summaries of one summarizer x prompt.

Scans <input_dir> recursively. Every folder that contains
    *_original.csv                  (step 2, --report-type original)
    *_<edit>_easy.csv, *_<edit>_medium.csv, *_<edit>_hard.csv   (step 2, --report-type attack)
for edit = alter_terms and/or missing_facts gets one merged file per edit type, written into that
folder. The merged file is the easy file (all its columns) plus
    patient_friendly_summary                      from *_original.csv
    patient_friendly_summary_adversarial_medium   from *_<edit>_medium.csv
    patient_friendly_summary_adversarial_hard     from *_<edit>_hard.csv
joined one-to-one on the report identifier (--join-key). This is the input of the judge (step 4).

Report identifiers: "study id" (Merlin, default), note_id (INSPECT), study_id (MIMIC-IV).

Usage:
  python pipeline/merge_patient_friendly_summaries.py summaries/inspect/medgemma/long --join-key note_id \
      --out-name-alter reports_medgemma_with_adv_alter.csv \
      --out-name-missing reports_medgemma_with_adv_missing.csv
"""

import argparse
import os
import sys
import glob
import pandas as pd

DIFFICULTIES = ["easy", "medium", "hard"]
# Report identifier used to join the files; set from --join-key in main().
#   Merlin: "study id"   INSPECT: "note_id"   MIMIC-IV: "study_id"
JOIN_KEYS = ["study id"]
ORIG_COL = "patient_friendly_summary"
ADV_COL_SRC = "patient_friendly_summary_adversarial"

def _fail(msg: str):
    print(f"ERROR: {msg}", file=sys.stderr); sys.exit(1)

def _find_one(pattern):
    m = glob.glob(pattern, recursive=False)
    return m[0] if m else None

def _dir_has_category(dirpath: str, category: str) -> bool:
    # Need original + all three difficulty files present
    if _find_one(os.path.join(dirpath, "*_original.csv")) is None:
        return False
    for d in DIFFICULTIES:
        if not glob.glob(os.path.join(dirpath, f"*_{category}_{d}.csv")):
            return False
    return True

def _align_key_dtypes(df_ref: pd.DataFrame, df: pd.DataFrame, path: str):
    for k in JOIN_KEYS:
        if k not in df.columns:
            _fail(f"Missing join key '{k}' in {path}")
        if df[k].dtype != df_ref[k].dtype:
            try:
                df[k] = df[k].astype(df_ref[k].dtype)
            except Exception as e:
                _fail(f"Cannot cast key '{k}' in {path} to {df_ref[k].dtype}: {e}")

def _merge_one_dir(dirpath: str, category: str) -> pd.DataFrame:
    """
    Base = <category>_easy.csv (keeps all its columns, including
    patient_friendly_summary_adversarial_easy).
    Adds:
      - patient_friendly_summary                     from *_original.csv
      - patient_friendly_summary_adversarial_medium  from *_<category>_medium.csv
      - patient_friendly_summary_adversarial_hard    from *_<category>_hard.csv
    """
    # Load base: EASY
    easy_path = _find_one(os.path.join(dirpath, f"*_{category}_easy.csv"))
    if easy_path is None:
        _fail(f"Missing {category} 'easy' CSV in {dirpath}")
    base = pd.read_csv(easy_path)

    for k in JOIN_KEYS:
        if k not in base.columns:
            _fail(f"Missing join key '{k}' in {easy_path} (set --join-key for this dataset)")
    if f'{ADV_COL_SRC}_easy' not in base.columns:
        _fail(f"Missing '{ADV_COL_SRC}_easy' in {easy_path}")

    # Bring in ORIGINAL's patient_friendly_summary (only that column + keys)
    orig_path = _find_one(os.path.join(dirpath, "*_original.csv"))
    if orig_path is None:
        _fail(f"Missing original CSV in {dirpath}")
    orig_df = pd.read_csv(orig_path)

    for k in JOIN_KEYS:
        if k not in orig_df.columns:
            _fail(f"Missing join key '{k}' in {orig_path}")
    if ORIG_COL not in orig_df.columns:
        _fail(f"Missing '{ORIG_COL}' in {orig_path}")

    _align_key_dtypes(base, orig_df, orig_path)
    orig_df = orig_df.loc[:, JOIN_KEYS + [ORIG_COL]]
    merged = base.merge(orig_df, on=JOIN_KEYS, how="left", validate="one_to_one")

    # Add MEDIUM and HARD adversarial columns (select only keys+adv col, then rename)
    for diff in ["medium", "hard"]:
        path = _find_one(os.path.join(dirpath, f"*_{category}_{diff}.csv"))
        if path is None:
            _fail(f"Missing {category} '{diff}' CSV in {dirpath}")
        df = pd.read_csv(path)
        _align_key_dtypes(merged, df, path)

        if f'{ADV_COL_SRC}_{diff}' not in df.columns:
            _fail(f"Missing '{ADV_COL_SRC}_{diff}' in {path}")

        df = df.loc[:, JOIN_KEYS + [f'{ADV_COL_SRC}_{diff}']]

        # Ensure we won't collide with existing non-key columns
        overlaps = (set(df.columns) - set(JOIN_KEYS)) & set(merged.columns)
        if overlaps:
            _fail(f"Unexpected overlapping columns {overlaps} from {path}")

        merged = merged.merge(df, on=JOIN_KEYS, how="left", validate="one_to_one")

    return merged

def main():
    global JOIN_KEYS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_dir", help="Root folder to scan recursively.")
    ap.add_argument("--join-key", default="study id",
                    help='Report identifier column: "study id" (Merlin, default), note_id (INSPECT), '
                         'study_id (MIMIC-IV).')
    ap.add_argument("--out-name-alter", default="merged_alter_terms_with_adversarials.csv",
                    help="Output filename for alter_terms (written into each subfolder).")
    ap.add_argument("--out-name-missing", default="merged_missing_facts_with_adversarials.csv",
                    help="Output filename for missing_facts (written into each subfolder).")
    args = ap.parse_args()
    JOIN_KEYS = [args.join_key]

    root = os.path.abspath(args.input_dir)
    if not os.path.isdir(root):
        _fail(f"Not a directory: {root}")

    # Candidate dirs are those that at least have *_original.csv; we’ll then validate category presence.
    candidates = {os.path.dirname(p) for p in glob.glob(os.path.join(root, "**", "*_original.csv"), recursive=True)}
    if not candidates:
        _fail("No subfolders with *_original.csv found.")

    processed = 0
    for d in sorted(candidates):
        did_any = False
        if _dir_has_category(d, "alter_terms"):
            df = _merge_one_dir(d, "alter_terms")
            outp = os.path.join(d, args.out_name_alter)
            df.to_csv(outp, index=False)
            print(f"[alter_terms] {d} -> {outp}  ({len(df)} rows)")
            did_any = True

        if _dir_has_category(d, "missing_facts"):
            df = _merge_one_dir(d, "missing_facts")
            outp = os.path.join(d, args.out_name_missing)
            df.to_csv(outp, index=False)
            print(f"[missing_facts] {d} -> {outp}  ({len(df)} rows)")
            did_any = True

        if did_any:
            processed += 1

    if processed == 0:
        _fail("Found *_original.csv files, but no complete (easy/medium/hard) sets for alter_terms or missing_facts.")
    else:
        print(f"Done. Wrote outputs for {processed} subfolder(s).")

if __name__ == "__main__":
    main()
