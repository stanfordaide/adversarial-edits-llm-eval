r"""
Step 5 of the pipeline: parse judge scores and build one wide score file per folder.

Run once per judge x dataset x prompt x edit type, on a folder holding the step-4 outputs
    llm_as_judge_<edit>_<difficulty>_<summarizer>_summaries_<judge>_judge.csv
for all summarizers and difficulties. For every file:
  1. Scores the judge script could not parse (empty factual_faithfulness, sensitivity_to_changes,
     certainty_fidelity or hallucination) are recovered from raw_judge_output with a more
     tolerant three-pass parser (label + value, "Score:" lines after a heading, next-line value).
  2. Brackets, asterisks and quotes are stripped from the `alterations` column.
Then all files are joined on the report identifier into one wide CSV (--combined-out) with the
dataset's ID columns, the ground-truth edits and, for each input file,
    <col>_<edit>_<difficulty>_<summarizer>_<judge>
for col in the four scores, eval_reasoning, raw_judge_output and alterations. These score files
are the inputs of the scripts in analysis/.

IMPORTANT: the folder path must contain
  - the dataset: "merlin", "inspect", or "mimic-iv" / "mimic_iv"  (selects the ID columns), and
  - the edit type: "alter_terms" or "missing_facts"               (selects the ground-truth columns),
e.g. judge/gpt_5/mimic-iv/long/alter_terms/. Name the combined file
    <judge>_<dataset>_<long|short>_<edit>_combined_cleaned_wide.csv
with <dataset> in {merlin, inspect, mimic}; the analysis scripts parse these names. A relative
--combined-out is written inside the folder.

Usage:
  python pipeline/clean_judge_outputs.py judge/gpt_5/merlin/long/alter_terms --glob "*_judge.csv" \
      --combined-out gpt_5_merlin_long_alter_terms_combined_cleaned_wide.csv
"""
import argparse, os, re, sys, glob
import pandas as pd

TARGET_COLS = ["factual_faithfulness", "sensitivity_to_changes", "certainty_fidelity", "hallucination"]

LABELS = {
    # accept optional "Category N:" prefix; keep original label spellings flexible
    "factual_faithfulness": r"(?:category\s*\d+\s*:\s*)?factual\s*faith(?:ful|full)?ness",
    # also accept 'Sensitivity to Omissions', 'Omission Sensitivity', and 'Sensitivity to Missing Facts/Information'
    "sensitivity_to_changes": (
        r"(?:category\s*\d+\s*:\s*)?"
        r"(?:"
            r"sensitivity\s*(?:to\s*)?(?:"
                r"change(?:s)?"
                r"|omission(?:s)?"
                r"|missing[\s\-]*(?:facts?|information)"
            r")"
            r"|(?:omission(?:s)?|missing[\s\-]*(?:facts?|information))\s*sensitivity"
        r")"
    ),
    "certainty_fidelity": r"(?:category\s*\d+\s*:\s*)?certainty\s*fidelity",
    # also accept "Hallucination Rate"
    "hallucination": r"(?:category\s*\d+\s*:\s*)?hallucination(?:s)?(?:\s*rate)?",
}

# Accept:
# - optional colon/hyphen
# - optional newline(s)
# - optional "Score:" label
# - optional Markdown bold **digit**
# - optional [digit]
# Capture only the leftmost number; allow optional slash-list after it (e.g., [2/1/2])
# --- VALUE PATTERN (string and compiled) ---

VALUE_PAT_STR = (
    r"(?:\s*[:\-]?\s*|\s*\n+)?"
    r"(?:score\s*[:\-]?\s*)?"
    r"\*{0,2}\[?\s*(\d{1,2})(?:\s*/[^\]\n]*)?\s*\]?"  # group 1 = first/left-most
    r"(?:\s+(\d{1,2}))?"                               # group 2 = optional trailing number
    r"\*{0,2}"
)

# Standalone value pattern (if you ever need it directly)
VALUE_PAT = re.compile(VALUE_PAT_STR, flags=re.I)

# Label + value patterns (for Pass 1)
COMPILED = {
    k: re.compile(LABELS[k] + VALUE_PAT_STR, flags=re.I | re.S)
    for k in LABELS
}

# Fallback: after finding a label, look ahead locally for a "Score: [x/y/z] 1" style line
SECTION_SCORE_PAT = re.compile(
    r"score\s*[:\-]\s*\*{0,2}\[?\s*(\d{1,2})(?:\s*/[^\]\n]*)?\s*\]?"  # group 1 = left-most
    r"(?:\s+(\d{1,2}))?"                                              # group 2 = optional trailing
    r"\*{0,2}",
    flags=re.I,
)


def extract_metric_scores(text: str):
    """Parse the four metric scores from a raw judge response. Returns {metric: int} (may be partial)."""
    out = {}
    if not isinstance(text, str) or not text.strip():
        return out

    # Pass 1: direct label + value (same vicinity)
    for col, pat in COMPILED.items():
        m = pat.search(text)
        if m:
            try:
                # trailing number wins if present, otherwise left-most in the brackets
                val_str = m.group(2) or m.group(1)
                v = int(val_str)
                if 0 <= v <= 10:
                    out[col] = v
            except Exception:
                pass


    # Pass 2: windowed "find Score after label" (handles headings like "Category X: ...\nScore: **0**")
    if len(out) < 4:
        for col, label_pat in LABELS.items():
            if col in out:
                continue
            for m in re.finditer(label_pat, text, flags=re.I):
                # search within a local window after the label
                start = m.end()
                window = text[start:start + 600]  # small, safe window
                ms = SECTION_SCORE_PAT.search(window)
                if ms:
                    try:
                        val_str = ms.group(2) or ms.group(1)
                        v = int(val_str)
                        if 0 <= v <= 10:
                            out[col] = v
                            break
                    except Exception:
                        pass

    # Pass 3: very light fallback (label then next-line number, allowing optional brackets/bold)
    if len(out) < 4:
        for col, label in LABELS.items():
            if col in out:
                continue
            m = re.search(
                label + r".{0,80}\n\s*(?:score\s*[:\-]\s*)?\*{0,2}\[?\s*(\d{1,2})\s*\]?\*{0,2}",
                text,
                flags=re.I
            )
            if m:
                try:
                    v = int(m.group(1))
                    if 0 <= v <= 10:
                        out[col] = v
                except Exception:
                    pass

    return out

# Only remove [, ], *, and " from 'alterations' content
STRIP_UNNECESSARY_RE = re.compile(r'[\[\]\*"]+')

def clean_alterations_keep_content(cell_text: str) -> str:
    if not isinstance(cell_text, str):
        return cell_text
    return STRIP_UNNECESSARY_RE.sub("", cell_text)

# ---- filename meta parser ----
# The edit/difficulty groups may split "alter_terms_easy" differently, but the suffix is the
# underscore-joined concatenation, e.g. "alter_terms_easy_claude_gpt_5".
FNAME_RE = re.compile(
    r"""^llm_as_judge_                # prefix
        (?P<edit>[^_]+)_              # edit
        (?P<difficulty>[^_]+)_        # difficulty
        (?P<summarizer>.+?)_          # summarizer (non-greedy, may contain underscores)
        summaries_
        (?P<judge>.+?)_               # judge (non-greedy, may contain dots/underscores)
        judge\.csv$                   # suffix
    """,
    flags=re.I | re.X
)

def parse_suffix_from_filename(fname: str) -> str | None:
    """Column suffix <edit>_<difficulty>_<summarizer>_<judge> from a judge output file name, or None."""
    base = os.path.basename(fname)
    m = FNAME_RE.match(base)
    if not m:
        return None
    edit = m.group("edit")
    difficulty = m.group("difficulty")
    summarizer = m.group("summarizer")
    judge = m.group("judge")
    # Keep underscores; normalize any dots to underscores for cleaner column names
    suffix = f"{edit}_{difficulty}_{summarizer}_{judge}".replace(".", "_")
    return suffix

# ---- DYNAMIC BASE COLS (by folder & edit type) ----
DEFAULT_BASE = [
    "person_id", "study id", "note_id",
    "gender_source_value", "birth_DATETIME",
    "race_source_value", "ethnicity_source_value", "gender_source_value",
]

def compute_base_cols(folder_path: str) -> list:
    """ID and ground-truth columns to carry into the wide file, chosen from the folder path."""
    p = folder_path.lower()

    # Ground-truth columns depend on the task
    if "alter_terms" in p:
        gt = [
            "ground_truth_alter_terms_easy",
            "ground_truth_alter_terms_medium",
            "ground_truth_alter_terms_hard",
        ]
    elif "missing_facts" in p:
        gt = [
            "ground_truth_missing_facts_easy",
            "ground_truth_missing_facts_medium",
            "ground_truth_missing_facts_hard",
        ]
    else:
        gt = []

    # Dataset-specific base IDs
    if "mimic-iv" in p or "mimic_iv" in p:
        print("MIMIC-IV")
        base = ["study_id", "subject_id", "hadm_id"] + gt
    elif "merlin" in p:
        base = ["study id"] + gt
    elif "inspect" in p:
        base = [
            "person_id", "study id", "note_id",
            "gender_source_value", "birth_DATETIME",
            "race_source_value", "ethnicity_source_value", "gender_source_value",
        ] + gt
    else:
        base = DEFAULT_BASE + gt

    # De-duplicate while preserving order
    seen = set()
    return [c for c in base if not (c in seen or seen.add(c))]

# Initialize with default; main() will override based on the folder path
BASE_COLS_WANTED = DEFAULT_BASE

# Columns we will replicate per file with suffix
REPLICATE_COLS = TARGET_COLS + ["eval_reasoning", "raw_judge_output", "alterations"]


def process_df(df: pd.DataFrame):
    """Fill missing score cells from raw_judge_output and lightly clean `alterations`."""
    for c in TARGET_COLS:
        if c not in df.columns:
            df[c] = pd.NA
    if "raw_judge_output" in df.columns:
        parsed = df["raw_judge_output"].apply(extract_metric_scores)
        for col in TARGET_COLS:
            miss = df[col].isna() | (df[col].astype(str).str.strip() == "")
            df.loc[miss, col] = parsed.apply(lambda d: d.get(col, pd.NA))
    if "alterations" in df.columns:
        df["alterations"] = df["alterations"].apply(clean_alterations_keep_content)
    return df


def safe_read_csv(path):
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path, engine="python")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="Folder with the judge output CSVs (path must name the dataset and edit type)")
    ap.add_argument("--glob", default="*_judge.csv", help="Glob pattern to match CSVs (default: *_judge.csv)")
    ap.add_argument("--inplace", action="store_true",
                    help="With --combined-out '': overwrite the input files instead of writing "
                         "*_cleaned_with_scores.csv")
    ap.add_argument("--combined-out", default="combined_cleaned_wide.csv",
                    help="Combined wide CSV (relative paths are written inside the folder). Pass '' to "
                         "write per-file cleaned CSVs instead")
    args = ap.parse_args()

    folder = os.path.abspath(args.folder)
    # Override base columns based on folder contents (inspect/missing_facts/mimic-iv/merlin)
    global BASE_COLS_WANTED
    BASE_COLS_WANTED = compute_base_cols(folder)

    files = sorted(glob.glob(os.path.join(folder, args.glob)))
    if not files:
        print(f"No files matched in {folder} with pattern {args.glob}", file=sys.stderr)
        sys.exit(1)

    total_updates = 0
    combined = None
    # Merge keys (same dataset detection as compute_base_cols)
    folder_lc = folder.lower()
    if "mimic-iv" in folder_lc or "mimic_iv" in folder_lc:
        KEY_PREF = ["study_id"]
    else:
        # Merlin, INSPECT and fallback: whichever of these ID columns are present
        KEY_PREF = ["person_id", "study id", "note_id"]

    for f in files:
        try:
            df = safe_read_csv(f)
        except Exception as e:
            print(f"[SKIP] {os.path.basename(f)}: cannot read CSV ({e})")
            continue

        # clean/process per-file
        before = {c: df[c].isna().sum() if c in df.columns else None for c in TARGET_COLS}
        df = process_df(df)
        after = {c: df[c].isna().sum() if c in df.columns else None for c in TARGET_COLS}

        updated = 0
        for c in TARGET_COLS:
            if before[c] is not None and after[c] is not None:
                updated += int(before[c] - after[c])
        total_updates += updated

        out_path = f if args.inplace else os.path.splitext(f)[0] + "_cleaned_with_scores.csv"
        try:
            # only save if not combined
            if not args.combined_out:
                df.to_csv(out_path, index=False)
                tag = "(in-place)" if args.inplace else "(new file)"
                print(f"[OK ] {os.path.basename(f)} -> {os.path.basename(out_path)} {tag}; filled {updated} score cells; alterations lightly cleaned.")
        except Exception as e:
            print(f"[FAIL] {os.path.basename(f)}: could not save ({e})")

        # ---- append to combined wide ----
        suffix = parse_suffix_from_filename(os.path.basename(f))
        if suffix is None:
            print(f"[WARN] {os.path.basename(f)} does not match expected pattern; skipping combine.")
            continue

        key_cols = [c for c in KEY_PREF if c in df.columns]
        if not key_cols:
            print(f"[WARN] {os.path.basename(f)}: no merge keys ({KEY_PREF}) found; skipping combine.")
            continue

        present_base = [c for c in BASE_COLS_WANTED if c in df.columns]
        sub = df[key_cols + [c for c in present_base if c not in key_cols]].copy()

        for c in REPLICATE_COLS:
            if c in df.columns:
                sub[f"{c}_{suffix}"] = df[c]

        if combined is None:
            combined = sub
        else:
            base_to_add = [c for c in present_base if c not in combined.columns and c not in key_cols]
            replicate_cols = [col for col in sub.columns
                              if any(col.startswith(f"{c}_") for c in REPLICATE_COLS)]
            keep_cols = key_cols + base_to_add + replicate_cols
            sub = sub[keep_cols]

            combined = pd.merge(combined, sub, on=key_cols, how="outer")

    # Write combined wide file
    if combined is not None:
        key_cols_all = [c for c in KEY_PREF if c in combined.columns]
        present_base = [c for c in BASE_COLS_WANTED if c in combined.columns and c not in key_cols_all]
        other_cols = [c for c in combined.columns if c not in key_cols_all + present_base]
        combined = combined[key_cols_all + present_base + other_cols]

        try:
            combined.to_csv(os.path.join(folder, args.combined_out), index=False)
            print(f"[OK ] Combined wide CSV -> {args.combined_out}")
        except Exception as e:
            print(f"[FAIL] Could not save combined CSV ({e})")
    else:
        print("[WARN] No combined CSV produced (no matching files/keys).")

    print(f"Done. Total score cells filled across files: {total_updates}")


if __name__ == "__main__":
    # unbuffered prints are handy when piping; optional
    try:
        main()
    except BrokenPipeError:
        # avoid stacktrace when piping to head/less
        pass