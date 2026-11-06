r"""
Step 4 of the pipeline (Azure variant): score patient-friendly summaries with an LLM judge.

This is the judge of llm_as_judge.py (same prompts, rubric, inputs and outputs; see that file's
docstring) with an additional `--model-type azure` backend for Azure OpenAI. The paper used it
for the GPT-5 judge on MIMIC-IV, because the PhysioNet data-use agreement requires the reports
to be processed through Azure OpenAI rather than the public OpenAI API.

Differences from llm_as_judge.py:
  - azure backend: the deployment (--azure-deployment, else AZURE_OPENAI_DEPLOYMENT, else
    "gpt-5") is sent as the model name; the key is read from AZURE_API_KEY and the endpoint from
    --azure-endpoint or AZURE_OPENAI_ENDPOINT.
  - an empty model response is retried once (all OpenAI-compatible backends; Ollama too).
  - --pre-facts is joined on study_id only (MIMIC-IV, Merlin), not on note_id.
  - like the openai judge, the azure judge does not extract findings from the summaries.

Usage:
  python pipeline/llm_as_judge_azure.py --model-type azure --model gpt-5 --azure-deployment gpt-5 \
      --attack-type alter_terms --difficulty easy --input <reports_claude_with_adv_alter.csv> \
      --output llm_as_judge_alter_terms_easy_claude_summaries_gpt_5_judge.csv
"""
import pandas as pd
import re
import time
import os
from openai import OpenAI, AzureOpenAI
import argparse
from generate_patient_friendly import extract_impression
from tqdm import tqdm

# Model configuration (defaults; --model-type / --model and the CLI overrides below take precedence).
# cost_per_1k_tokens is only used for the printed cost estimate (USD).
MODEL_CONFIG = {
    "nvidia": {
        "model": "deepseek-ai/deepseek-r1",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "api_key": os.environ.get("NVIDIA_API_KEY"),
        "max_tokens": 8096,
        "sleep_time": 7.5,
        "extra_params": {"top_p": 0.9, "extra_body": {"showThinking": False}}
    },
    "openai": {
        "model": "gpt-5",
        "reasoning_effort": "low",
        "verbosity": "low",
        "base_url": None,
        "api_key": os.environ.get("OPENAI_API_KEY"),
        "extra_params": {},
        "sleep_time": 0,
        "cost_per_1k_tokens": {"input": 0.00125, "output": 0.01}  # GPT-5 list price: $1.25/M in, $10/M out
    },
    "ollama": {
        "model": os.environ.get("OLLAMA_MODEL", "llama3.1"),
        "base_url": os.environ.get("OLLAMA_HOST", "http://localhost:1112"),
        "api_key": None,
        "max_tokens": 4000,
        "sleep_time": 0,
        "extra_params": {},
        "cost_per_1k_tokens": {"input": 0, "output": 0}  # Local model, no cost
    },
    # vLLM (OpenAI-compatible server), used for the MedGemma judge
    "vllm": {
        "model": os.environ.get("VLLM_MODEL", "google/medgemma-27b-text-it"),
        "base_url": os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
        "api_key": os.environ.get("VLLM_API_KEY", "sk-noop"),
        "max_tokens": 4000,
        "sleep_time": 0,
        "extra_params": {},  # keep empty; vLLM accepts standard OpenAI params
        "cost_per_1k_tokens": {"input": 0, "output": 0}
    },
    # Azure OpenAI, used for the GPT-5 judge on MIMIC-IV
    "azure": {
        "deployment": os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5"),
        "endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT"),
        "api_version": os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        "api_key": os.environ.get("AZURE_API_KEY"),
        "sleep_time": 0,
        "extra_params": {},               # keep empty; standard OpenAI params are fine
        "cost_per_1k_tokens": {"input": 0.00125, "output": 0.01}  # estimate only; actual Azure rates may differ
    },
}

# Cost tracking
total_tokens = {"input": 0, "output": 0}
total_cost = 0.0

# Import-time default only; when run as a script, --model-type overrides this.
MODEL_TYPE = os.environ.get("MODEL_TYPE", "ollama")

# Initialize client based on model type
config = MODEL_CONFIG[MODEL_TYPE]
if MODEL_TYPE == "ollama":
    import requests
    
    # Clean up host if it includes protocol
    if '://' in config['base_url']:
        config['base_url'] = config['base_url'].split('://')[-1]
    
    def ollama_model_call(prompt, model=config["model"]):
        try:
            # Ensure the base_url has the http:// protocol
            base_url = config['base_url']
            if not base_url.startswith('http://'):
                base_url = f'http://{base_url}'
            
            response = requests.post(
                f"{base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False
                }
            )
            
            if response.status_code != 200:
                print(f"Ollama API error: {response.status_code}")
                return None
                
            response_json = response.json()
            return response_json.get("response", "")
        except Exception as e:
            print(f"Ollama API error:", e)
            return None


# Strip <think>...</think> blocks
# for QWEN and deepseek
def strip_think(text):
    if not isinstance(text, str):
        return text
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

# for medgemma model
def strip_unused(text):
    if not isinstance(text, str):
        return text
    return re.sub(r"<unused94>.*?<unused95>", "", text, flags=re.DOTALL).strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-type", choices=["nvidia", "openai", "ollama", "vllm", "azure"], default="ollama",
                      help="Judge backend (nvidia, openai, ollama, vllm, azure). Default: ollama")
    parser.add_argument("--model", default="llama3.1",
                        help="Judge model name (e.g. gpt-5). For azure the deployment name is sent instead")
    parser.add_argument("--ollama-host", default=None, help="Host for Ollama server (overrides OLLAMA_HOST; default localhost)")
    parser.add_argument("--ollama-port", default=None, help="Port for Ollama server (default 1112)")
    parser.add_argument("--attack-type", choices=["alter_terms", "missing_facts"], default="alter_terms",
                      help="Type of attack to evaluate")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="hard",
                      help="Difficulty level for attack reports")
    parser.add_argument("--input", required=True, help="Merged summary CSV from step 3")
    parser.add_argument("--output", required=True,
                        help="Output CSV; name it llm_as_judge_<attack>_<difficulty>_<summarizer>_summaries_<judge>_judge.csv")
    parser.add_argument("--vllm-base-url", default=None, help="Base URL for vLLM OpenAI server (overrides VLLM_BASE_URL env var)")
    parser.add_argument("--pre-facts", default=None,
                        help="Previous judge output CSV for the same reports, attack type and difficulty, with "
                             "facts_original and facts_adversarial columns (skips report fact extraction). "
                             "Joined on study_id.")
    # Optional Azure overrides (env vars still work if you skip these)
    parser.add_argument("--azure-endpoint", default=None, help="Azure OpenAI endpoint (https://<res>.cognitiveservices.azure.com/)")
    parser.add_argument("--azure-deployment", default=None, help="Azure deployment name (used as model in calls)")
    parser.add_argument("--azure-api-version", default=None, help="Azure OpenAI API version (e.g., 2024-12-01-preview)")

    args = parser.parse_args()

    SELECTED_MODEL = args.model
    # Update configuration based on command line arguments
    MODEL_TYPE = args.model_type
    MODEL_CONFIG["ollama"]["model"] = args.model
    MODEL_CONFIG["openai"]["model"] = args.model
    
    # Azure overrides (optional)
    if args.azure_endpoint:
        MODEL_CONFIG["azure"]["endpoint"] = args.azure_endpoint
    if args.azure_deployment:
        MODEL_CONFIG["azure"]["deployment"] = args.azure_deployment
    if args.azure_api_version:
        MODEL_CONFIG["azure"]["api_version"] = args.azure_api_version

    
    # Handle Ollama host configuration
    if args.ollama_host or args.ollama_port:
        # If either host or port is provided via command line, construct new base_url
        host = args.ollama_host or "localhost"
        port = args.ollama_port or "1112"
        MODEL_CONFIG["ollama"]["base_url"] = f"http://{host}:{port}"
    else:
        # Use the environment variable as is
        MODEL_CONFIG["ollama"]["base_url"] = os.environ.get("OLLAMA_HOST", "http://localhost:1112")
    
    # vLLM base URL override (optional)
    if args.vllm_base_url:
        MODEL_CONFIG["vllm"]["base_url"] = args.vllm_base_url
        MODEL_CONFIG["vllm"]["model"] = args.model
    # Initialize client based on model type
    config = MODEL_CONFIG[MODEL_TYPE]
    if MODEL_TYPE == "ollama":
        import requests
        import json
        
        def ollama_model_call(prompt, model=config["model"]):
            try:
                base_url = config['base_url']
                if not base_url.startswith('http://'):
                    base_url = f'http://{base_url}'

                def _once():
                    import requests
                    r = requests.post(
                        f"{base_url}/api/generate",
                        json={"model": model, "prompt": prompt, "stream": False}
                    )
                    if r.status_code != 200:
                        return None
                    j = r.json()
                    return j.get("response", "")

                out = _once()
                if not out or not isinstance(out, str) or not out.strip():
                    time.sleep(0.4)  # tiny backoff
                    out = _once()
                return out
            except Exception as e:
                print("Ollama API error:", e)
                return None
    elif MODEL_TYPE == "azure":
        # Azure OpenAI client initialization
        azure_endpoint = args.azure_endpoint or MODEL_CONFIG["azure"]["endpoint"]
        azure_api_key = MODEL_CONFIG["azure"]["api_key"]
        azure_api_version = args.azure_api_version or MODEL_CONFIG["azure"]["api_version"]

        print(f"AzureOpenAI client -> endpoint={azure_endpoint}, deployment={MODEL_CONFIG['azure']['deployment']}")

        client = AzureOpenAI(
            api_version=azure_api_version,
            azure_endpoint=azure_endpoint,
            api_key=azure_api_key,
        )
    else:
        print(f"OpenAI-compatible client -> base_url={MODEL_CONFIG[MODEL_TYPE]['base_url'] or 'default'}, model={SELECTED_MODEL}")
        client = OpenAI(
            base_url=config["base_url"],
            api_key=config["api_key"],
        )

    # Load your CSV file
    df = pd.read_csv(args.input)
    
    # --- OPTIONAL: load precomputed facts and merge on study_id ---
    # Stage-1 facts are extracted from the original and adversarial REPORTS, so they are identical
    # for every summarizer. In the paper they were extracted once per attack type x difficulty (in the
    # first judge run) and reused for the remaining summarizers by passing that run's output here.
    pre_facts_path = args.pre_facts

    if pre_facts_path and os.path.exists(pre_facts_path):
        try:
            df_facts = pd.read_csv(pre_facts_path)
            # Normalize merge keys
            if "study_id" not in df.columns and "study id" in df.columns:
                df["study_id"] = df["study id"]
            if "study_id" not in df_facts.columns and "study id" in df_facts.columns:
                df_facts = df_facts.rename(columns={"study id": "study_id"})

            if "study_id" in df.columns and "study_id" in df_facts.columns:
                keep_cols = ["study_id"]
                for c in ["facts_original", "facts_adversarial"]:
                    if c in df_facts.columns:
                        keep_cols.append(c)
                
                df = df.merge(df_facts[keep_cols], on="study_id", how="left")
            else:
                print("[WARN] Could not merge precomputed facts (missing 'study_id' in one of the files). Skipping.")
        except Exception as e:
            print(f"[WARN] Failed to load/merge precomputed facts from {pre_facts_path}: {e}")
    
    df["simplified_original"] = df["patient_friendly_summary"]
    df["simplified_adversarial"] = df[f"patient_friendly_summary_adversarial_{args.difficulty}"]
    
    mask_think = df["patient_friendly_summary"].str.startswith("<think>", na=False)
    mask_unused = df["patient_friendly_summary"].str.startswith("<unused94>", na=False)

    df.loc[mask_think, "simplified_original"] = df.loc[mask_think, "patient_friendly_summary"].apply(strip_think)
    df.loc[mask_unused, "simplified_original"] = df.loc[mask_unused, "patient_friendly_summary"].apply(strip_unused)

    mask_think = df[f"patient_friendly_summary_adversarial_{args.difficulty}"].str.startswith("<think>", na=False)
    mask_unused = df[f"patient_friendly_summary_adversarial_{args.difficulty}"].str.startswith("<unused94>", na=False)

    df.loc[mask_think, "simplified_adversarial"] = df.loc[mask_think, f"patient_friendly_summary_adversarial_{args.difficulty}"].apply(strip_think)
    df.loc[mask_unused, "simplified_adversarial"] = df.loc[mask_unused, f"patient_friendly_summary_adversarial_{args.difficulty}"].apply(strip_unused)
    
    # Prepare output columns
    output_cols = [
        "facts_simplified_original", "facts_simplified_adversarial",
        "factual_faithfulness", "sensitivity_to_changes", "certainty_fidelity", "hallucination", "eval_reasoning",
        "raw_judge_output"  # New column for raw judge output
    ]
    for col in output_cols:
        df[col] = None

    # Prompt for fact extraction
    def build_fact_extraction_prompt(title, text):
        return f"""You are a clinical NLP specialist. Your task is to extract structured findings from the following radiology summary.

Each finding must be reported using the following fixed schema:
- Finding ID: (start from 1, increment by 1)
- Condition: (the medical condition being described, e.g., "appendicitis")
- Location: (the anatomical location, if any, e.g., "appendix", "lungs"; use "not specified" if missing)
- Severity: (e.g., "mild", "moderate", "severe"; use "not applicable" if not stated)
- Certainty: (must be exactly one of: "no evidence", "possible", "likely", "confirmed", "uncertain")
- Differential: (if a differential diagnosis is mentioned, otherwise "not applicable")

Important Instructions:
- If a field is not mentioned, fill it with the default (e.g., "not specified", "not applicable").
- Certainty must be strictly one of the five listed values.
- Do not infer differentials unless explicitly stated.
- If there are no findings (e.g., "normal study"), respond: "No findings."
- Use exactly the output layout below. No explanations or comments.

Output format:

Finding #1:
Condition:
Location:
Severity:
Certainty:
Differential (if any):

Finding #2: ...

Here is the radiology summary:
{title}:
{text}
"""

    # Judge prompts: term substitution (alter_terms) and impression removal (missing_facts)
    def build_judge_prompt_with_facts_v2(facts_A, facts_B, simp_A, simp_B, ground_truth_alterations):
        return f"""You are evaluating whether Simplified B reflects the factual differences between Original Report A and B.You are a medical NLP evaluation expert. Your task is to compare extracted findings from four versions of a radiology summary:
1. Original radiology report (Extracted Findings: {facts_A} )
2. Adversarially modified radiology report (Extracted Findings: {facts_B})
3. Patient-friendly summary of the original report (Simplified A: {simp_A})
4. Patient-friendly summary of the adversarial report (Simplified B: {simp_B})

You are given:
- Extracted structured findings from each version
- Ground-truth alterations applied to create the adversarial report: {ground_truth_alterations}

Your task is to evaluate the *patient-friendly adversarial summary* (Simplified B) by comparing it to the other three versions and the ground-truth alterations. 

Score the summary on the following four categories:
---

### Category 1: Factual Faithfulness  
Did Simplified B preserve all unaltered facts from Simplified A correctly?

→ Score: 0 (inaccurate), 1 (fully faithful)

### Category 2: Sensitivity to Changes  
Did Simplified B reflect the adversarial term substitutions appropriately?

→ Score: 0 (missed), 1 (partially), or 2 (clearly reflected)

### Category 3: Certainty Fidelity  
Did Simplified B correctly convey the diagnostic uncertainty (e.g., "no evidence" vs. "possible")?

→ Score: 0 (increased certainty), 1 (preserved or matched)

### Category 4: Hallucination Rate  
Did Simplified B introduce any findings or details not present in any of the 3 other versions?

→ Score: 0 (hallucinated), 1 (no hallucinations)

---

### Format your output strictly like this:

Factual Faithfulness: [0/1]
Sensitivity to Changes: [0/1/2]
Certainty Fidelity: [0/1]
Hallucination: [0/1]
Alterations: <e.g., ["acute -> healed", "obstruction -> stenosis"]>
Reasoning: 
factual_faithfulness_reasoning: <explanation here>
sensitivity_to_changes_reasoning: <explanation here>
certainty_fidelity_reasoning: <explanation here>
hallucination_reasoning: <explanation here>
  
Important Notes:
- Use the extracted findings, not surface language alone.
- Focus on the *meaning* of alterations, not just word changes.
- Hallucination = introducing any new clinical findings not found in any of the 3 source documents.
- Certainty changes must be clinically meaningful (e.g., "possible pneumonia" vs. "pneumonia" matters).

"""

    def build_judge_prompt_with_missing_facts(facts_A, facts_B, simp_A, simp_B, ground_truth_removals):
        return f"""You are a medical NLP evaluation expert. Your task is to compare extracted findings from four versions of a radiology summary:
1. Original radiology report (Extracted Findings: {facts_A} )
2. Adversarially modified radiology report (Extracted Findings: {facts_B})
3. Patient-friendly summary of the original report (Simplified A: {simp_A})
4. Patient-friendly summary of the adversarial report (Simplified B: {simp_B})

You are given:
- Extracted structured findings from each version
- Ground-truth alterations, which list the facts or sentences that were REMOVED from the original report: {ground_truth_removals}

Your task is to evaluate the *patient-friendly adversarial summary* (Simplified B) by comparing it to the other three versions and the ground-truth removals. 

Score the summary on the following four categories:
---

### Category 1: Factual Faithfulness  
Did Simplified B preserve all unaltered facts from Simplified A correctly?

→ Score: 0 (inaccurate), 1 (fully faithful)

### Category 2: Sensitivity to Omissions  
Did Simplified B correctly omit the findings that were removed in the adversarial report?

→ Score:
  - 0 (Missed): The summary incorrectly includes the finding that was supposed to be removed.
  - 1 (Partially Reflected): The summary omits the main finding but leaves behind a confusing remnant or ambiguously refers to it.
  - 2 (Clearly Reflected): The summary correctly and clearly omits the removed finding.

### Category 3: Certainty Fidelity  
Did Simplified B correctly convey the diagnostic uncertainty (e.g., "no evidence" vs. "possible") for the remaining facts?

→ Score: 0 (increased certainty), 1 (preserved or matched)

### Category 4: Hallucination Rate  
Did Simplified B introduce any findings or details not present in any of the 3 other versions?

→ Score: 0 (hallucinated), 1 (no hallucinations)

---

### Format your output strictly like this:

Factual Faithfulness: [0/1]
Sensitivity to Changes: [0/1/2]
Certainty Fidelity: [0/1]
Hallucination: [0/1]
Alterations: <e.g., ["Removed: Acute appendicitis without evidence of abscess."]>
Reasoning: 
factual_faithfulness_reasoning: <explanation here>
sensitivity_to_changes_reasoning: <explanation here>
certainty_fidelity_reasoning: <explanation here>
hallucination_reasoning: <explanation here>
  
Important Notes:
- Use the extracted findings, not surface language alone.
- Focus on the *meaning* of the omissions, not just word count.
- Hallucination = introducing any new clinical findings not found in any of the 3 source documents.
- Certainty changes must be clinically meaningful (e.g., "possible pneumonia" vs. "pneumonia" matters).

"""

    def _first_nonempty_text(completion):
        """Return (text, finish_reason). text is None if nothing usable."""
        try:
            choices = getattr(completion, "choices", []) or []
            finish = None
            for ch in choices:
                finish = getattr(ch, "finish_reason", finish)
                msg = getattr(ch, "message", None)
                if not msg:
                    continue
                content = getattr(msg, "content", None)
                if isinstance(content, str) and content.strip():
                    return content.strip(), getattr(ch, "finish_reason", finish)
            return None, finish
        except Exception:
            return None, None

    # Model call wrapper
    def model_call(prompt, model=None):
        """Unified model call wrapper for OpenAI, NVIDIA, vLLM, and Ollama."""
        try:
            # Always fall back to the selected model from args
            if model is None:
                model = args.model

            if MODEL_TYPE == "ollama":
                return ollama_model_call(prompt, model)

            # Build OpenAI-compatible request
            params = {
                "model": model,
                "reasoning_effort": "low",
                "verbosity": "low",
                "messages": [{"role": "user", "content": prompt + "\n\nEnsure all sections are filled out fully."}],
                **config.get("extra_params", {})
            }

            # Azure: use deployment name up front
            if MODEL_TYPE == "azure":
                params["model"] = (
                getattr(args, "azure_deployment", None)
                or MODEL_CONFIG["azure"].get("deployment")
                )

            completion = client.chat.completions.create(**params)

            # Token + cost tracking for OpenAI only
            if MODEL_TYPE == "openai":
                global total_tokens, total_cost
                input_tokens = getattr(getattr(completion, "usage", None), "prompt_tokens", 0)
                output_tokens = getattr(getattr(completion, "usage", None), "completion_tokens", 0)
                total_tokens["input"] += input_tokens
                total_tokens["output"] += output_tokens
                input_cost = (input_tokens / 1000) * config["cost_per_1k_tokens"]["input"]
                output_cost = (output_tokens / 1000) * config["cost_per_1k_tokens"]["output"]
                total_cost += input_cost + output_cost
                print(f"Last call tokens - Input: {input_tokens}, Output: {output_tokens}")
                print(f"Last call cost: ${(input_cost + output_cost):.4f}")

            # Extract text robustly
            text = None
            finish = None
            if isinstance(completion, str):
                text = completion
            elif hasattr(completion, "choices"):
                text, finish = _first_nonempty_text(completion)

            # If empty/None, try exactly once more (cheap + fast)
            if not text:
                # Optional: cap tokens on retry to save cost if your model supports it
                params_retry = dict(params)
                if "max_completion_tokens" not in params_retry:
                    params_retry["max_completion_tokens"] = 1512
                else:
                    try:
                        params_retry["max_completion_tokens"] = min(1512, params_retry["max_completion_tokens"])
                    except Exception:
                        params_retry["max_completion_tokens"] = 1512

                # Azure uses deployment name as model; keep it the same
                if MODEL_TYPE == "azure":
                    params_retry["model"] = MODEL_CONFIG["azure"]["deployment"]

                time.sleep(0.4)  # tiny backoff
                completion2 = client.chat.completions.create(**params_retry)

                # Token + cost tracking for OpenAI on retry as well
                if MODEL_TYPE == "openai":
                    input_tokens = getattr(getattr(completion2, "usage", None), "prompt_tokens", 0)
                    output_tokens = getattr(getattr(completion2, "usage", None), "completion_tokens", 0)
                    total_tokens["input"] += input_tokens
                    total_tokens["output"] += output_tokens
                    input_cost = (input_tokens / 1000) * config["cost_per_1k_tokens"]["input"]
                    output_cost = (output_tokens / 1000) * config["cost_per_1k_tokens"]["output"]
                    total_cost += input_cost + output_cost
                    print(f"(retry) Last call tokens - Input: {input_tokens}, Output: {output_tokens}")
                    print(f"(retry) Last call cost: ${(input_cost + output_cost):.4f}")

                text2 = None
                if isinstance(completion2, str):
                    text2 = completion2
                elif hasattr(completion2, "choices"):
                    text2, _ = _first_nonempty_text(completion2)

                if text2 and isinstance(text2, str) and text2.strip():
                    return text2

            # Return text if we have it, else None
            if text and isinstance(text, str):
                return text
            return None

        except Exception as e:
            print(f"{MODEL_TYPE.upper()} API error:", e)
            return None

    # Main loop: rows 0-100 only. Summaries exist for the first 100 rows (step 2), and rows
    # without summaries are skipped below, so this scores the 100-report evaluation subset.
    # Each scored row is appended to the output file (the header is written with row 0).
    output_file = args.output
    for idx, row in tqdm(df.iterrows()):
        if idx > 100:
            break
        # Check for missing or NaN values in required fields
        required_fields = [f"adversarial_report_{args.attack_type}_{args.difficulty}", 
                          "patient_friendly_summary", f"patient_friendly_summary_adversarial_{args.difficulty}"]        
        if any(pd.isna(row[field]) or row[field] == "" for field in required_fields):
            if "study id" in row:
                print(f"\nSkipping report_id {row['study id']} - Missing required data")
            elif "study_id" in row:
                print(f"\nSkipping report_id {row['study_id']} - Missing required data")
            continue
        
        # check if row[f"adversarial_report_{args.attack_type}_{args.difficulty}"] starts with <think> and if yes strip it
        if row[f"adversarial_report_{args.attack_type}_{args.difficulty}"].startswith("<think>"):
            row[f"adversarial_report_{args.attack_type}_{args.difficulty}"] = strip_think(row[f"adversarial_report_{args.attack_type}_{args.difficulty}"])
        elif row[f"adversarial_report_{args.attack_type}_{args.difficulty}"].startswith("<unused94>"):
            row[f"adversarial_report_{args.attack_type}_{args.difficulty}"] = strip_unused(row[f"adversarial_report_{args.attack_type}_{args.difficulty}"])
        
        if "study id" in row:
            print(f"\nEvaluating report_id {row['study id']}...")
        elif "study_id" in row:
            print(f"\nEvaluating report_id {row['study_id']}...")

        # 1. Extract facts
        # if "impressions" in row then dont extract the impression
        if "impressions" in row:
            row["original_impression"] = row["impressions"]
        else:
            row["original_impression"] = extract_impression(row["Findings"])

        # Prefer precomputed facts if available (merged from CSV). Fallback to model_call otherwise.
        if "facts_original" in df.columns and pd.notna(row.get("facts_original", None)) and str(row["facts_original"]).strip():
            facts_A = str(row["facts_original"]).strip()
        else:
            facts_A = model_call(build_fact_extraction_prompt("Original Report A", row["original_impression"]))

        if "facts_adversarial" in df.columns and pd.notna(row.get("facts_adversarial", None)) and str(row["facts_adversarial"]).strip():
            facts_B = str(row["facts_adversarial"]).strip()
        else:
            facts_B = model_call(build_fact_extraction_prompt("Original Report B", row[f"adversarial_report_{args.attack_type}_{args.difficulty}"]))
        
        if MODEL_TYPE not in ('openai', 'azure'):
            facts_simp_A = model_call(build_fact_extraction_prompt("Simplified A", row["simplified_original"]))
            facts_simp_B = model_call(build_fact_extraction_prompt("Simplified B", row["simplified_adversarial"]))
        else:
            facts_simp_A = "No simplified fact extraction requested."
            facts_simp_B = "No simplified fact extraction requested."
        # Strip <think> tags from all fact extractions
        if MODEL_TYPE == "nvidia":
            facts_A = strip_think(facts_A)
            facts_B = strip_think(facts_B)
            facts_simp_A = strip_think(facts_simp_A)
            facts_simp_B = strip_think(facts_simp_B)
            
        # for medgemma model
        if facts_A.startswith("<unused94>"):
            facts_A = strip_unused(facts_A)
        if facts_B.startswith("<unused94>"):
            facts_B = strip_unused(facts_B)
        if facts_simp_A.startswith("<unused94>"):
            facts_simp_A = strip_unused(facts_simp_A)
        if facts_simp_B.startswith("<unused94>"):
            facts_simp_B = strip_unused(facts_simp_B)

        df.at[idx, "facts_original"] = facts_A
        df.at[idx, "facts_adversarial"] = facts_B
        df.at[idx, "facts_simplified_original"] = facts_simp_A
        df.at[idx, "facts_simplified_adversarial"] = facts_simp_B

        # 2. Score quality based on fact differences
        if args.attack_type == "alter_terms":
            judge_prompt = build_judge_prompt_with_facts_v2(
                facts_A, 
                facts_B, 
                row["simplified_original"], 
                row["simplified_adversarial"], 
                row[f"ground_truth_{args.attack_type}_{args.difficulty}"]
        )
        elif args.attack_type == "missing_facts":
            judge_prompt = build_judge_prompt_with_missing_facts(
                facts_A, 
                facts_B, 
                row["simplified_original"], 
                row["simplified_adversarial"], 
                row[f"ground_truth_{args.attack_type}_{args.difficulty}"]
            )

        judge_result = model_call(judge_prompt)

        if judge_result:
            if MODEL_TYPE == "nvidia":
                # Strip <think> tags from judge result
                judge_result = strip_think(judge_result)
                
            if judge_result.startswith("<unused94>"):
                judge_result = strip_unused(judge_result)
            # print(judge_result)    
            
            # Save raw judge output
            df.at[idx, "raw_judge_output"] = judge_result
            
            # More flexible regex patterns that handle asterisks and variations in category names
            ff = re.search(r"Factual Faithfulness(?: Rate)?:\s*[*]*\s*(\d+)\s*[*]*", judge_result)
            sens = re.search(r"Sensitivity to Changes(?: Rate)?:\s*[*]*\s*(\d+)\s*[*]*", judge_result)
            cert = re.search(r"Certainty Fidelity(?: Rate)?:\s*[*]*\s*(\d+)\s*[*]*", judge_result)
            hall = re.search(r"Hallucination(?: Rate)?:\s*[*]*\s*(\d+)\s*[*]*", judge_result)
            alterations = re.search(r"Alterations:\s*(.*?)(?:\n|$)", judge_result)
            reason = re.search(r"Reasoning:\s*(.*)", judge_result, re.DOTALL)

            # Extract numbers from asterisks if present
            def extract_number(match):
                if not match:
                    return None
                num = match.group(1)
                return int(num.strip('*'))

            df.at[idx, "factual_faithfulness"] = extract_number(ff)
            df.at[idx, "sensitivity_to_changes"] = extract_number(sens)
            df.at[idx, "certainty_fidelity"] = extract_number(cert)
            df.at[idx, "hallucination"] = extract_number(hall)
            df.at[idx, "alterations"] = alterations.group(1).strip() if alterations else None
            df.at[idx, "eval_reasoning"] = strip_think(reason.group(1).strip()) if reason else None

        # Save the current row to a temporary file
        current_row = df.iloc[[idx]]
        if idx == 0:
            current_row.to_csv(output_file, mode='w', index=False)
        else:
            current_row.to_csv(output_file, mode='a', header=False, index=False)
        
        # print(f"Saved progress for report_id {row['study id']}")
        # if MODEL_TYPE == "openai":
        #     print(f"\nCumulative totals:")
        #     print(f"Total tokens - Input: {total_tokens['input']}, Output: {total_tokens['output']}")
        #     print(f"Total cost so far: ${total_cost:.4f}")
        
        time.sleep(config["sleep_time"])

    print("\nEvaluation complete. Results saved to", output_file)
    if MODEL_TYPE == "openai":
        print(f"\nFinal usage statistics:")
        print(f"Total tokens used - Input: {total_tokens['input']}, Output: {total_tokens['output']}")
        print(f"Total cost: ${total_cost:.4f}")
