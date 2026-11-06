# Adversarial Evaluation of LLMs for Patient-Friendly Radiology Report Summarization

Code for *Adversarial Evaluation of Large Language Models for Patient-Friendly Radiology
Report Summarization*. It covers:

1. generation of adversarial radiology impressions from a term dictionary (term substitution
   and impression removal);
2. patient-friendly summary generation with six LLMs and two prompts;
3. LLM-as-a-judge scoring (GPT-5 primary, MedGemma secondary);
4. the statistical analyses reported in the paper and supplement.

**What is included:** the full pipeline, the analysis code, the adversarial term dictionary
(`dictionary/`) and the per-dataset medical-term lists (`data/medical_terms/`).

**What is not included:** the radiology reports, the generated summaries and judge outputs
(including the final score files), the radiologist reader-study ratings, and the code used to
build the dictionary and to draw the figures. The reports must be obtained from their original
sources (see [Data](#data)); everything downstream of them can be regenerated with this code.

## Contents

```
LICENSE                                         MIT license
requirements.txt                                pinned package versions used for the paper
dictionary/
  adversarial_term_dictionary_v1_as_used.json   dictionary used for every result in the paper
  adversarial_term_dictionary_v2_cleaned.json   corrected version recommended for new work
  CHANGELOG.md                                  every difference between v1 and v2
data/medical_terms/
  medical_terms_{inspect,merlin,mimic_mri}.json  per-dataset medical term lists used to rank
                                                 impression points for impression removal
pipeline/
  adversarial_report_generation.py    step 1  adversarial impressions
  generate_patient_friendly.py        step 2  patient-friendly summaries
  merge_patient_friendly_summaries.py step 3  merge original + adversarial summaries
  llm_as_judge.py                     step 4  LLM judge (OpenAI, vLLM, Ollama backends)
  llm_as_judge_azure.py               step 4  LLM judge, adds Azure OpenAI (used for MIMIC-IV)
  clean_judge_outputs.py              step 5  parse scores and build the per-folder score file
analysis/
  summary_statistics.py              Figures 3a/3b/4b: means, Wilson CIs, hallucination rates,
                                     Detailed-vs-Concise permutation tests
  sensitivity_statistics.py          Figure 4a: sensitivity-score distributions + permutation tests
  failure_mode_analysis.py           Figure 5: category- and term-level miss rates
  file_order_{gpt_5,medgemma}.txt    score-file order that reproduces the published p-values
  multifactor_models/                GEE / proportional-odds models, omnibus tests, concordance
                                     with the permutation tests
  judge_validity/                    chance-corrected judge-radiologist agreement (reader study)
```

Every script has a module docstring describing its inputs and outputs; run it with `--help`
to see that docstring and all options.

## Setup

Python 3.10, the version used for the paper. The pinned versions in `requirements.txt` target
3.10 and may not install on much newer Python versions.

```bash
pip install -r requirements.txt
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_md-0.5.4.tar.gz
```

- **Analysis only** (`analysis/`): numpy, pandas, scipy, statsmodels, patsy and scikit-learn
  are enough. Installing `irrCAC` (optional) enables an extra cross-check in
  `agreement_analysis.py`.
- **Pipeline** (`pipeline/`): needs the full requirements. Steps 1 and 4 import the impression
  extractor from `generate_patient_friendly.py`, which in turn imports `torch`, `transformers`
  and `openai`, so those packages are needed even for the steps that do not call a model.
  Importing it also creates a log file, `radiology_processing.log`, in the working directory.

Run all commands from the repository root (default paths such as the dictionary are relative
to it).

**Credentials are read from environment variables only.** Nothing is stored in the code.

| Used by | Variable(s) |
|---|---|
| OpenAI summarizer / GPT-5 judge | `OPENAI_API_KEY` |
| Claude summarizer | `ANTHROPIC_API_KEY` |
| Azure summarizer (`generate_patient_friendly.py --model-type azure`) | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT` (or `--azure-endpoint`) |
| Azure judge (`llm_as_judge_azure.py --model-type azure`) | `AZURE_API_KEY`, `AZURE_OPENAI_ENDPOINT` (or `--azure-endpoint`), optionally `AZURE_OPENAI_DEPLOYMENT`, `AZURE_OPENAI_API_VERSION` |
| vLLM backend | `VLLM_BASE_URL` (or `--vllm-base-url`), optionally `VLLM_API_KEY` |
| Ollama backend | `--ollama-host` / `--ollama-port`, or `OLLAMA_HOST` (summarizer default `localhost:11434`; judge default `localhost:1112`) |

Note that the Azure summarizer and the Azure judge read the key from different variables
(`AZURE_OPENAI_API_KEY` vs `AZURE_API_KEY`).

## Data

The radiology reports are **not** redistributed. Obtain them from the original sources:

| Dataset | Reports | Split used | Text column |
|---|---|---|---|
| Merlin (abdominal CT) | 5,125 | public test split | `Findings` (impression extracted from it) |
| INSPECT (chest CTPA) | 3,676 | predefined test split | `impressions` |
| MIMIC-IV (MRI reports; requires PhysioNet credentialing) | 5,000 | rows marked `test` in an added `Split` column | `Findings` (impression extracted from it) |

Every input CSV needs a `Split` column; step 1 keeps only rows with `Split == "test"`. Each
dataset also needs its report identifier, which is used to join files in later steps:
`study id` (Merlin), `note_id` (INSPECT) or `study_id` (MIMIC-IV).

The generated summaries, judge outputs and score files contain report text and are **not**
distributed. In particular, files derived from MIMIC-IV must not be shared outside the
PhysioNet data-use agreement. `.gitignore` excludes `data/judge_scores/` and
`data/reader_study/` so they are not committed by accident.

## Pipeline

| Step | Script | Input | Output |
|---|---|---|---|
| 1 | `adversarial_report_generation.py` | report CSV | `<input>_with_adversarial_<edit>.csv` |
| 2 | `generate_patient_friendly.py` | step-1 CSV | one summary CSV per model × prompt × report type × difficulty |
| 3 | `merge_patient_friendly_summaries.py` | folder of step-2 CSVs | one merged CSV per edit type |
| 4 | `llm_as_judge.py` / `llm_as_judge_azure.py` | merged CSV | one judge CSV per dataset × prompt × edit type × difficulty × summarizer × judge |
| 5 | `clean_judge_outputs.py` | folder of judge CSVs | one score file per judge × dataset × prompt × edit type |

Edit types are named `alter_terms` (term substitution) and `missing_facts` (impression removal)
throughout the code.

### 1. Adversarial impressions

```bash
python pipeline/adversarial_report_generation.py --dataset merlin --strategy alter_terms \
    --input_csv <merlin_reports.csv>
python pipeline/adversarial_report_generation.py --dataset merlin --strategy missing_facts \
    --input_csv <merlin_reports.csv>
```

`--dataset` is `merlin`, `inspect` or `mimic`. Each run writes `<input>_with_adversarial_<strategy>.csv`
next to the input, containing the test rows with `adversarial_report_<strategy>_<difficulty>`
and `ground_truth_<strategy>_<difficulty>` columns for difficulty easy / medium / hard.

- **Term substitution** replaces dictionary terms with an alternative from the same entry
  (easy: 1 term in 1 category; medium: up to 2 terms in each of 2 categories; hard: up to 4 in
  each of 4), preferring the least-used terms. Each measurement (e.g. "1.5 cm") is also replaced
  with a random value and unit with 50% probability.
- **Impression removal** removes the numbered impression points with the most medical terms
  (easy: 1; medium: up to 2; hard: up to 3, never more than half). Single-point impressions
  fall back to removing sentences, and single sentences to removing one medical word.
- `--term_alternatives_config` defaults to the v1 dictionary used in the paper.
- The medical-term list defaults to `data/medical_terms/medical_terms_<dataset>.json`
  (`medical_terms_mimic_mri.json` for `mimic`).
- The paper's adversarial reports were generated **without a random seed**, so re-running
  produces different substitutions. `--seed` makes new runs repeatable.

### 2. Patient-friendly summaries

One run per model × prompt × report type × difficulty. The **detailed** prompt is the
default; add `--short-prompt` for the **concise** prompt.

```bash
# original report (writes column patient_friendly_summary)
python pipeline/generate_patient_friendly.py --input <..._with_adversarial_alter_terms.csv> \
    --output summaries/merlin/claude/long/merlin_claude_original.csv \
    --model-type claude --model claude-sonnet-4-5 --report-type original
# adversarial report (writes column patient_friendly_summary_adversarial_<difficulty>)
python pipeline/generate_patient_friendly.py --input <..._with_adversarial_alter_terms.csv> \
    --output summaries/merlin/claude/long/merlin_claude_alter_terms_easy.csv \
    --model-type claude --model claude-sonnet-4-5 --report-type attack --difficulty easy
```

| Summarizer | `--model-type` | `--model` |
|---|---|---|
| GPT-5-mini | `openai` (`azure` for MIMIC-IV) | `gpt-5-mini` |
| Claude Sonnet 4.5 | `claude` | `claude-sonnet-4-5` |
| Llama 3.1 (8B) | `ollama` | `llama3.1:latest` |
| Llama 3.3 (70B) | `ollama` | `llama3.3:latest` |
| MedGemma 27B | `ollama` | `medgemma3:27b` |
| Qwen 3 (4B) | `ollama` | `qwen3:4b` |

**Evaluation subset:** summaries are generated for the **first 100 rows of each input file**,
in file order. These 100 reports per dataset (300 in total) are the ones scored in the paper.

**Output layout expected by step 3.** Put the outputs for one dataset × summarizer × prompt in
their own folder, with exactly one file per name pattern below (the prefix is free):

```
summaries/<dataset>/<model>/<long|short>/
  *_original.csv              --report-type original (one file, shared by both edit types)
  *_alter_terms_easy.csv      --report-type attack --difficulty easy, input <...>_with_adversarial_alter_terms.csv
  *_alter_terms_medium.csv    ... --difficulty medium
  *_alter_terms_hard.csv      ... --difficulty hard
  *_missing_facts_easy.csv    --report-type attack --difficulty easy, input <...>_with_adversarial_missing_facts.csv
  *_missing_facts_medium.csv
  *_missing_facts_hard.csv
```

Step 3 needs all three difficulties for an edit type. The analyses use easy/medium/hard for
term substitution and easy/hard for impression removal; medium impression-removal scores are
dropped if present. Each run also writes a checkpoint, `<output>.partial`, which can be deleted
afterwards.

### 3. Merge summaries

```bash
python pipeline/merge_patient_friendly_summaries.py summaries/merlin/<model>/<long|short> \
    --join-key "study id" \
    --out-name-alter reports_<model>_with_adv_alter.csv \
    --out-name-missing reports_<model>_with_adv_missing.csv
```

`--join-key` is the report identifier: `"study id"` (Merlin, the default), `note_id` (INSPECT)
or `study_id` (MIMIC-IV), so run the merge separately per dataset. The folder is scanned
recursively (e.g. `summaries/merlin` merges every model and prompt at once), and the merged files
are written into every subfolder that has a complete set.

### 4. LLM judge

```bash
python pipeline/llm_as_judge.py --model-type openai --model gpt-5 \
    --attack-type alter_terms --difficulty easy \
    --input summaries/merlin/claude/long/reports_claude_with_adv_alter.csv \
    --output judge/gpt_5/merlin/long/alter_terms/llm_as_judge_alter_terms_easy_claude_summaries_gpt_5_judge.csv
```

The judge works in two stages. It first extracts structured findings from the original and
adversarial impressions. Non-OpenAI judges also extract findings from the two summaries; these
are saved but not used for scoring. It then scores the adversarial summary against the
extracted findings, the two summaries and the ground-truth edits:

| Metric | Score | Meaning |
|---|---|---|
| Factual faithfulness | 0 / 1 | unaltered facts preserved |
| Sensitivity to changes | 0 / 1 / 2 | edit missed / partially / fully reflected |
| Certainty fidelity | 0 / 1 | diagnostic certainty preserved |
| Hallucination | 0 / 1 | **1 = no** new finding introduced |

- **GPT-5 judge:** `--model-type openai --model gpt-5`. For MIMIC-IV, use
  `llm_as_judge_azure.py --model-type azure --model gpt-5 --azure-deployment gpt-5`
  (PhysioNet data-use requirements).
- **MedGemma judge:** `--model-type vllm --model google/medgemma-27b-text-it`.
- **Fact reuse (`--pre-facts`):** the facts extracted from the original and adversarial
  *reports* are identical for every summarizer. Extract them once per dataset × attack type ×
  difficulty, then pass that first output file to the remaining runs to skip re-extraction. Files are joined
  on `study_id` (Merlin's `study id` is normalized to it) or, in `llm_as_judge.py` only, on
  `note_id` (INSPECT).
- The judge processes rows 0–100 of its input and skips rows without summaries, so it scores the
  same 100 reports. The raw judge response is kept in `raw_judge_output` and parsed again in step 5.
- **Output naming matters.** Name files
  `llm_as_judge_<alter_terms|missing_facts>_<easy|medium|hard>_<summarizer>_summaries_<judge>_judge.csv`,
  with summarizer ∈ {claude, gpt, llama_3_1, llama_3_3, medgemma, qwen} and judge ∈ {gpt_5, medgemma}.
  Step 5 builds column names from these filenames.

### 5. Parse scores and build score files

Put all judge outputs (6 summarizers × difficulties) for one judge × dataset × prompt × edit type
in one folder, then run:

```bash
python pipeline/clean_judge_outputs.py judge/gpt_5/merlin/long/alter_terms --glob "*_judge.csv" \
    --combined-out gpt_5_merlin_long_alter_terms_combined_cleaned_wide.csv
```

- **The folder path must name the dataset and the edit type.** The script picks the ID and
  ground-truth columns from substrings of the path: `merlin`, `inspect` or `mimic-iv`/`mimic_iv`,
  and `alter_terms` or `missing_facts`.
- Name the combined file `<judge>_<dataset>_<long|short>_<alter_terms|missing_facts>_combined_cleaned_wide.csv`
  with `<dataset>` ∈ {merlin, inspect, mimic} (`mimic`, not `mimic-iv`). The analysis scripts parse
  these names. A relative `--combined-out` is written inside the folder.
- Scores missing from the judge's own parse are recovered from `raw_judge_output` with a more
  tolerant parser.

Run this once per judge × dataset × prompt × edit type. That gives 12 score files per judge,
which are the inputs to the analyses.

## Reproducing the analyses

Copy the 12 score files of each judge from step 5 into `data/judge_scores/<judge>/`. Then:

```bash
# Figures 3a, 3b, 4b (+ the tidy table used by the next scripts)
python analysis/summary_statistics.py --scores-dir data/judge_scores/gpt_5 --judge gpt_5 \
    --file-order analysis/file_order_gpt_5.txt --out-dir results/gpt_5
# Figure 4a
python analysis/sensitivity_statistics.py --tidy results/gpt_5/all_datasets_tidy_gpt_5.csv \
    --out-dir results/gpt_5
# Figure 5
python analysis/failure_mode_analysis.py --scores-dir data/judge_scores/gpt_5 --judge gpt_5 \
    --dictionary dictionary/adversarial_term_dictionary_v1_as_used.json --out-dir results/gpt_5
```

Replace `gpt_5` with `medgemma` for the supplementary MedGemma-judge results. Each script's
docstring (`--help`) lists the tables it writes and the figure each one belongs to.

### Multi-factor models and judge-radiologist agreement

- `analysis/multifactor_models/` fits multi-factor models to the GPT-5 judge scores: GEE
  logistic regression for the binary metrics and a proportional-odds model for sensitivity,
  clustered on report. It reports omnibus Type III tests, interaction tests, odds ratios and
  marginal means, then checks that the model-based prompt effects agree with the published
  permutation tests.
- `analysis/judge_validity/` re-analyses the radiologist reader study (105 cases) with
  prevalence-robust, chance-corrected agreement coefficients (Gwet's AC1/AC2, Brennan-Prediger),
  permutation tests against chance, and judge-vs-inter-radiologist comparisons.

```bash
python analysis/multifactor_models/build_model_dataset.py \
    --scores-dir data/judge_scores/gpt_5 --out-dir results/multifactor_models \
    --reference-tidy results/gpt_5/all_datasets_tidy_gpt_5.csv \
    --reference-hallucination-tests results/gpt_5/HallucinationRates_AllDatasets_gpt_5_long_vs_short_perm_tests.csv
python analysis/multifactor_models/fit_models.py \
    --data results/multifactor_models/model_dataset_gpt5_judge.csv --out-dir results/multifactor_models   # ~7 min
python analysis/multifactor_models/concordance_permutation.py \
    --results-dir results/gpt_5 --model-dir results/multifactor_models
python analysis/judge_validity/agreement_analysis.py \
    --reader-csv <reader_study_ratings.csv> --out-dir results/judge_validity
```

The reader-study ratings are not distributed. `agreement_analysis.py` expects one row per case
with, for each metric, the columns `<metric>_gpt_5`, `<metric>_rad_a`, `<metric>_rad_b` and
`<metric>_medgemma`. For hallucination, the radiologist columns are `hallucination_rate_rad_a`
and `hallucination_rate_rad_b`.

## Notes for reproduction

- **Score coding.** Every metric is scored so that 1 is the desirable outcome. That includes
  hallucination, where 1 = *no* hallucination, so hallucination rates count score == 0. Sensitivity
  to changes is 0/1/2 (not / partially / fully reflected).
- **Prompts.** In the score files and outputs, `Long` = detailed prompt and `Short` = concise prompt.
- **Re-running the pipeline is not deterministic.** Adversarial generation was unseeded and the
  LLM outputs vary between calls, so regenerated summaries and scores will differ from the
  paper's. Given the same score files, the analysis scripts are deterministic.
- **Permutation p-values** use 20,000 permutations with a fixed seed, but the random stream is
  consumed in row order. The `--file-order` files give the order of the paper's own score files,
  which reproduces the published p-values exactly. Without them, p-values differ only by Monte
  Carlo error.
- **Statistics are verbatim.** Every statistical function in the three figure scripts in
  `analysis/` is copied verbatim from the scripts that produced the paper's figures; only the
  plotting was removed. On the paper's score files they reproduce every published table exactly.
- **Summarizer keys in failure-mode output.** Summarizer keys parsed from GPT-5 score columns
  carry a `_gpt` tail (e.g. `claude_gpt`). `failure_mode_analysis.py` reports a cleaned
  `summarizer` column alongside the raw key.
- **Unused backends.** `generate_patient_friendly.py` and the judge scripts also support
  `nvidia` (NVIDIA NIM), and the summarizer supports `local` (Hugging Face `transformers`) and
  `vllm`. These were not used for the paper's results.

## License

This repository is released under the [MIT License](LICENSE). The radiology reports and anything
derived from them remain subject to the data-use terms of their original sources.

## Citation

If you use this code or the term dictionary, please cite:

> Paschali M, Aali A, Koirala A, Johnston A, McNamara K, Gaus R, Daneshjou R, Larson DB,
> Chaudhari AS. Adversarial Evaluation of Large Language Models for Patient-Friendly
> Radiology Report Summarization.
