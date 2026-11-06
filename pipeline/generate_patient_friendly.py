r"""
Step 2 of the pipeline: generate patient-friendly summaries of radiology impressions.

Each run summarizes ONE report type with ONE model and ONE prompt:
  --report-type original   summarizes the original impression, extracted from `Findings` with
                           extract_impression (or taken from `impressions` when there is no
                           `Findings` column; step 1 copies INSPECT's `impressions` into
                           `Findings`), and writes the column `patient_friendly_summary`
  --report-type attack     summarizes adversarial_report_<edit>_<difficulty> from step 1 and
                           writes `patient_friendly_summary_adversarial_<difficulty>`
                           (the edit type is taken from whichever column is present)

Prompts (identical across backends):
  detailed (default)  role, four focus points and the evaluation criteria
  concise             --short-prompt: "Create a patient-friendly summary of the following
                      radiology impression"
In the score files and analysis outputs these are labelled Long and Short.

Backends (--model-type): the paper used openai, azure (MIMIC-IV), claude and ollama; vllm,
nvidia (NVIDIA NIM) and local (Hugging Face transformers) are also available. API keys are read
from environment variables (see README) unless --api-key is given.

Only rows with Split in {test, val} are kept, and only the FIRST 100 rows are summarized (the
evaluation subset of the paper); the remaining rows are written with empty summary columns.
The output CSV keeps all input columns. A checkpoint `<output>.partial` is written after each
batch, and a log is appended to radiology_processing.log in the working directory.

extract_impression() is also imported by the step-1 and step-4 scripts.

Usage:
  python pipeline/generate_patient_friendly.py --input <..._with_adversarial_alter_terms.csv> \
      --output <out.csv> --model-type openai --model gpt-5-mini --report-type attack --difficulty easy
"""
import pandas as pd
import regex as re
import os
import time
import requests
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from openai import OpenAI, AzureOpenAI
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("radiology_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# --- Cost tracking --------------------------------------------------------

class CostTracker:
    """
    Tracks token usage & cost per provider/model.
    Print-only (no CSV). If API doesn't return usage, estimate tokens ~= len(text)/4.
    Prices are PER 1K TOKENS.
    """
    def __init__(self, in_price_per_1k=None, out_price_per_1k=None):
        self.default_in = float(in_price_per_1k) if in_price_per_1k is not None else None
        self.default_out = float(out_price_per_1k) if out_price_per_1k is not None else None
        self.totals = defaultdict(lambda: {"input_tokens": 0, "output_tokens": 0, "cost": 0.0})

    @staticmethod
    def _est_tokens(s: str) -> int:
        return max(1, int(len(s) / 4))  # coarse fallback

    def add(self, provider, model, prompt_text, completion_text,
            usage=None, in_price_per_1k=None, out_price_per_1k=None):
        pin = float(in_price_per_1k) if in_price_per_1k is not None else self.default_in or 0.0
        pout = float(out_price_per_1k) if out_price_per_1k is not None else self.default_out or 0.0

        if usage and ("prompt_tokens" in usage or "input_tokens" in usage):
            in_tok = usage.get("prompt_tokens", usage.get("input_tokens", 0))
            out_tok = usage.get("completion_tokens", usage.get("output_tokens", 0))
        else:
            in_tok = self._est_tokens(prompt_text or "")
            out_tok = self._est_tokens(completion_text or "")

        cost = (in_tok / 1000.0) * pin + (out_tok / 1000.0) * pout

        key = f"{provider}:{model}"
        self.totals[key]["input_tokens"] += in_tok
        self.totals[key]["output_tokens"] += out_tok
        self.totals[key]["cost"] += cost

    def summarize(self):
        total_cost = sum(v["cost"] for v in self.totals.values())
        print("\n===== Cost Summary =====")
        for k, v in self.totals.items():
            print(f"{k} | in_tok={v['input_tokens']} out_tok={v['output_tokens']} cost=${v['cost']:.4f}")
        print(f"TOTAL COST: ${total_cost:.4f}\n")
        return total_cost

# --- Summarizer backends ---------------------------------------------------
# Each processor exposes generate_summary(impression) -> str and builds the same two prompts.
# On failure it returns a string starting with "Error" instead of raising.

# vLLM (OpenAI-compatible server)
class VLLMLLMProcessor:
    def __init__(self, model_name="meta-llama/Llama-3-8B-Instruct", base_url=None, api_key=None, use_long_prompt=True):
        """
        Initialize processor for a running vLLM OpenAI-compatible server.

        Args:
            model_name: model name as served by vLLM
            base_url: base URL of vLLM OpenAI server (e.g., http://localhost:8000/v1)
            api_key: optional; only if your server enforces auth
            use_long_prompt: same semantics as others
        """
        self.model_name = model_name
        self.use_long_prompt = use_long_prompt
        self.base_url = base_url or os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
        # allow no key; many vLLM setups don't require one
        self.client = OpenAI(base_url=self.base_url, api_key=api_key or os.environ.get("VLLM_API_KEY", "EMPTY"))
        logger.info(f"Initialized vLLM client at {self.base_url} for model {model_name}")

    def _build_prompt(self, impression):
        if self.use_long_prompt:
            return f"""
                You are a medical communication specialist. Your task is to convert the following radiology impression into a clear, simple summary that a patient without medical background can understand.

                Focus on:
                1. Explaining medical terms in plain language
                2. Highlighting what these findings mean for the patient
                3. Avoiding unnecessarily alarming language
                4. Being honest but reassuring where appropriate

                You will be evaluated in terms of completeness, conciseness, accuracy, and existence of hallucinations.
                        
                Radiology Impression:
                {impression}
                
                Patient-Friendly Explanation:
                """
        else:
            return f"""Create a patient-friendly summary of the following radiology impression:
                {impression}
                """

    def generate_summary(self, impression):
        try:
            prompt = self._build_prompt(impression)
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                top_p=0.9,
                max_tokens=1000
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating summary with vLLM: {str(e)}")
            return f"Error generating summary: {str(e)}"


# OpenAI API
class OpenAIProprietaryProcessor:
    def __init__(self, model_name="gpt-5-mini", api_key=None, use_long_prompt=True,
                 cost_tracker=None, in_price_per_1k=None, out_price_per_1k=None):
        """
        Processor for OpenAI GPT models (API key from OPENAI_API_KEY unless given).
        Prices are only used for the printed cost estimate; defaults are gpt-5-mini list
        prices: input $0.25/M, output $2.00/M -> per 1K: 0.00025 / 0.00200.
        """
        self.model_name = model_name
        self.use_long_prompt = use_long_prompt
        self.client = OpenAI(api_key=(api_key or os.environ.get("OPENAI_API_KEY")))

        # hardcoded defaults; can be overridden via constructor
        self.cost_tracker = cost_tracker
        # gpt-nano
        # self.in_price = 0.00005 if in_price_per_1k is None else float(in_price_per_1k)
        # self.out_price = 0.00040 if out_price_per_1k is None else float(out_price_per_1k)
        # Per-1K-token pricing lookup; fall back to gpt-5-mini rates
        _pricing = {
            "gpt-5-mini":          (0.00025, 0.00200),
            "gpt-5.4-2026-03-05":  (0.0025,  0.015),
        }
        default_in, default_out = _pricing.get(model_name, (0.00025, 0.00200))
        self.in_price = default_in if in_price_per_1k is None else float(in_price_per_1k)
        self.out_price = default_out if out_price_per_1k is None else float(out_price_per_1k)
        logger.info(f"Initialized OpenAI client for model {model_name}")

    def _build_prompt(self, impression):
        if self.use_long_prompt:
            return f"""
                You are a medical communication specialist. Your task is to convert the following radiology impression into a clear, simple summary that a patient without medical background can understand.

                Focus on:
                1. Explaining medical terms in plain language
                2. Highlighting what these findings mean for the patient
                3. Avoiding unnecessarily alarming language
                4. Being honest but reassuring where appropriate

                You will be evaluated in terms of completeness, conciseness, accuracy, and existence of hallucinations.
                        
                Radiology Impression:
                {impression}
                
                Patient-Friendly Explanation:
                """
        else:
            return f"""Create a patient-friendly summary of the following radiology impression:
                {impression}
                """

    def generate_summary(self, impression):
        try:
            prompt = self._build_prompt(impression)
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            msg = completion.choices[0].message.content.strip()

            if self.cost_tracker:
                usage_obj = getattr(completion, "usage", None)
                usage = {}
                if usage_obj:
                    usage = {
                        "prompt_tokens": getattr(usage_obj, "prompt_tokens", 0),
                        "completion_tokens": getattr(usage_obj, "completion_tokens", 0),
                    }
                self.cost_tracker.add("openai", self.model_name, prompt, msg, usage,
                                      in_price_per_1k=self.in_price,
                                      out_price_per_1k=self.out_price)
            return msg
        except Exception as e:
            logger.error(f"Error generating summary with OpenAI: {str(e)}")
            return f"Error generating summary: {str(e)}"


# Azure OpenAI (used for MIMIC-IV)
class AzureOpenAIProcessor:
    def __init__(
        self,
        model_name="gpt-5-mini",                  # logical model name (optional)
        deployment=None,                          # Azure deployment name (used as model in calls)
        endpoint=None,                            # e.g., "https://<res>.cognitiveservices.azure.com/"
        api_version="2025-04-01-preview",
        api_key=None,                             # read from AZURE_OPENAI_API_KEY
        use_long_prompt=True,
        cost_tracker=None,
        in_price_per_1k=None,
        out_price_per_1k=None
    ):
        """
        Processor for Azure OpenAI Chat Completions.
        We use 'deployment' as the effective model argument (Azure semantics).
        Defaults pricing to match OpenAI gpt-5-mini placeholders unless overridden.
        """
        self.use_long_prompt = use_long_prompt
        self.deployment = deployment or model_name  # deployment acts as model on Azure
        self.client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT"),
            api_key=api_key or os.environ.get("AZURE_OPENAI_API_KEY"),
        )

        self.cost_tracker = cost_tracker
        # default per-1K token prices (override to your internal rates if needed)
        self.in_price = 0.00025 if in_price_per_1k is None else float(in_price_per_1k)
        self.out_price = 0.00200 if out_price_per_1k is None else float(out_price_per_1k)
        self.provider_label = "azure"

        logger.info(f"Initialized Azure OpenAI client for deployment '{self.deployment}'")

    def _build_prompt(self, impression):
        if self.use_long_prompt:
            return f"""
                You are a medical communication specialist. Your task is to convert the following radiology impression into a clear, simple summary that a patient without medical background can understand.

                Focus on:
                1. Explaining medical terms in plain language
                2. Highlighting what these findings mean for the patient
                3. Avoiding unnecessarily alarming language
                4. Being honest but reassuring where appropriate

                You will be evaluated in terms of completeness, conciseness, accuracy, and existence of hallucinations.
                        
                Radiology Impression:
                {impression}
                
                Patient-Friendly Explanation:
                """
        else:
            return f"""Create a patient-friendly summary of the following radiology impression:
                {impression}
                """

    def generate_summary(self, impression):
        try:
            prompt = self._build_prompt(impression)
            logger.info(f"Prompt: {prompt}")
            completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=4000,
                model=self.deployment,  # Azure: pass deployment name here
            )
            msg = completion.choices[0].message.content.strip()
            logger.info(msg)
            # Cost tracking
            if self.cost_tracker:
                usage_obj = getattr(completion, "usage", None)
                usage = {}
                if usage_obj:
                    usage = {
                        "prompt_tokens": getattr(usage_obj, "prompt_tokens", 0),
                        "completion_tokens": getattr(usage_obj, "completion_tokens", 0),
                    }
                self.cost_tracker.add(
                    self.provider_label,
                    self.deployment,
                    prompt,
                    msg,
                    usage,
                    in_price_per_1k=self.in_price,
                    out_price_per_1k=self.out_price,
                )
            return msg
        except Exception as e:
            logger.error(f"Error generating summary with Azure OpenAI: {str(e)}")
            return f"Error generating summary: {str(e)}"


class ClaudeLLMProcessor:
    def __init__(self, model_name="claude-4-5-sonnet-latest", api_key=None, use_long_prompt=True,
                 cost_tracker=None, in_price_per_1k=None, out_price_per_1k=None):
        """
        Processor for Anthropic Claude models using the Messages API (plain HTTPS request;
        API key from ANTHROPIC_API_KEY unless given).
        Prices are only used for the printed cost estimate; defaults are Claude Sonnet 4.5 list
        prices: input $3/M, output $15/M -> per 1K: 0.003 / 0.015.
        """
        self.model_name = model_name
        self.use_long_prompt = use_long_prompt
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.base_url = "https://api.anthropic.com/v1/messages"
        self.anthropic_version = os.environ.get("ANTHROPIC_VERSION", "2023-06-01")

        # hardcoded defaults; can be overridden
        self.cost_tracker = cost_tracker
        # sonnet 4.5
        self.in_price = 0.003 if in_price_per_1k is None else float(in_price_per_1k)
        self.out_price = 0.015 if out_price_per_1k is None else float(out_price_per_1k)
        logger.info(f"Initialized Claude client for model {model_name}")

    def _build_prompt(self, impression):
        if self.use_long_prompt:
            return f"""
                You are a medical communication specialist. Your task is to convert the following radiology impression into a clear, simple summary that a patient without medical background can understand.

                Focus on:
                1. Explaining medical terms in plain language
                2. Highlighting what these findings mean for the patient
                3. Avoiding unnecessarily alarming language
                4. Being honest but reassuring where appropriate

                You will be evaluated in terms of completeness, conciseness, accuracy, and existence of hallucinations.
                        
                Radiology Impression:
                {impression}
                
                Patient-Friendly Explanation:
                """
        else:
            return f"""Create a patient-friendly summary of the following radiology impression:
                {impression}
                """

    def generate_summary(self, impression):
        try:
            prompt = self._build_prompt(impression)
            headers = {
                "x-api-key": self.api_key if self.api_key else "",
                "anthropic-version": self.anthropic_version,
                "content-type": "application/json"
            }
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 64000
            }
            resp = requests.post(self.base_url, headers=headers, json=payload, timeout=60)
            if resp.status_code != 200:
                logger.error(f"Claude API error {resp.status_code}: {resp.text}")
                return f"Error generating summary: HTTP {resp.status_code}"

            data = resp.json()
            # Collect text blocks
            parts = []
            for block in data.get("content", []):
                if block.get("type") == "text" and "text" in block:
                    parts.append(block["text"])
            msg = "\n".join(parts).strip() if parts else ""

            # Usage -> cost
            if self.cost_tracker:
                usage = {}
                if "usage" in data:
                    usage = {
                        "input_tokens": data["usage"].get("input_tokens", 0),
                        "output_tokens": data["usage"].get("output_tokens", 0),
                    }
                self.cost_tracker.add("anthropic", self.model_name, prompt, msg, usage,
                                      in_price_per_1k=self.in_price,
                                      out_price_per_1k=self.out_price)
            return msg
        except Exception as e:
            logger.error(f"Error generating summary with Claude: {str(e)}")
            return f"Error generating summary: {str(e)}"

class NVIDIALLMProcessor:
    def __init__(self, model_name="deepseek-ai/deepseek-r1", api_key=None, use_long_prompt=True):
        """
        Initialize the LLM processor with NVIDIA R1 NIM.
        
        Args:
            model_name: NVIDIA model name to use
            api_key: API key for NVIDIA API (optional if running in NGC)
            use_long_prompt: Whether to use detailed prompt (True) or short prompt (False)
        """
        self.model_name = model_name
        self.use_long_prompt = use_long_prompt
        
        # Get API key from environment variable if not provided
        if api_key is None:
            api_key = os.environ.get("NVIDIA_API_KEY")
            
        logger.info(f"Initializing NVIDIA R1 NIM model: {model_name}")
        
        try:
            # Initialize OpenAI client with NVIDIA base URL
            self.client = OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=api_key
            )
            
            logger.info(f"NVIDIA R1 NIM client initialized successfully")
        
        except Exception as e:
            logger.error(f"Error initializing NVIDIA R1 NIM client: {str(e)}")
            raise

    def generate_summary(self, impression):
        """
        Generate a patient-friendly summary of the radiology impression using NVIDIA R1 NIM.
        """
        try:
            # Create the prompt based on length preference
            if self.use_long_prompt:
                prompt = f"""
                You are a medical communication specialist. Your task is to convert the following radiology impression into a clear, simple summary that a patient without medical background can understand.

                Focus on:
                1. Explaining medical terms in plain language
                2. Highlighting what these findings mean for the patient
                3. Avoiding unnecessarily alarming language
                4. Being honest but reassuring where appropriate

                You will be evaluated in terms of completeness, conciseness, accuracy, and existence of hallucinations.
                        
                Radiology Impression:
                {impression}
                
                Patient-Friendly Explanation:
                """
            else:
                prompt = f"""Create a patient-friendly summary of the following radiology impression:
                {impression}
                """
            
            # Generate text using NVIDIA R1 NIM with increased max_tokens
            logger.info(f"Generating summary for impression of length {len(impression)}")
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                top_p=0.9,
                max_tokens=1000  # Increased from 500 to 1000
            )
            
            # Return the complete, unfiltered model output
            summary = completion.choices[0].message.content.strip()
            logger.info(f"Generated summary of length {len(summary)}")
            # print(f"FULL MODEL OUTPUT:\n{summary}\n{'='*50}")
            
            return summary
        
        except Exception as e:
            logger.error(f"Error generating summary with NVIDIA R1 NIM: {str(e)}")
            return f"Error generating summary: {str(e)}"
    
    def __del__(self):
        """Clean up resources when done"""
        pass  # No specific cleanup needed for API client


class OpenSourceLLMProcessor:
    def __init__(self, model_name="medalpaca/medalpaca-7b", device=None, max_length=512, use_long_prompt=True):
        """
        Initialize the LLM processor with an open-source medical language model.
        
        Args:
            model_name: HuggingFace model name/path to use
            device: Device to run the model on (cuda, cpu, etc.)
            max_length: Maximum sequence length for the model
            use_long_prompt: Whether to use detailed prompt (True) or short prompt (False)
        """
        self.model_name = model_name
        self.max_length = max_length
        self.use_long_prompt = use_long_prompt
        
        # Determine device - use GPU if available, otherwise CPU
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Initializing model {model_name} on {self.device}")
        
        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Use lower precision for GPU to reduce memory usage
            if self.device == "cuda":
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name, 
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    device_map="auto"
                )
                # When using device_map="auto", don't specify device in pipeline
                self.generator = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    batch_size=8,  # Process multiple inputs at once
                    max_length=self.max_length,  # Limit sequence length
                    truncation=True  # Enable truncation
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(model_name)
                self.model.to(self.device)
                # For CPU, we can specify the device as -1
                self.generator = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=-1,
                    max_length=self.max_length,  # Limit sequence length
                    truncation=True  # Enable truncation
                )
            
            logger.info(f"Model {model_name} loaded successfully")
        
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def generate_summary(self, impression):
        """
        Generate a patient-friendly summary of the radiology impression.
        """
        try:
            if self.use_long_prompt:
                prompt = f"""
                You are a medical communication specialist. Your task is to convert the following radiology impression into a clear, simple summary that a patient without medical background can understand.

                Focus on:
                1. Explaining medical terms in plain language
                2. Highlighting what these findings mean for the patient
                3. Avoiding unnecessarily alarming language
                4. Being honest but reassuring where appropriate

                You will be evaluated in terms of completeness, conciseness, accuracy, and existence of hallucinations.
                        
                Radiology Impression:
                {impression}
                
                Patient-Friendly Explanation:
                """
            else:
                prompt = f"""Create a patient-friendly summary of the following radiology impression:
                {impression}
                """
            
            # Check token length and truncate if necessary
            input_ids = self.tokenizer.encode(prompt, truncation=True, max_length=self.max_length - 100)  # Leave room for generation
            truncated_prompt = self.tokenizer.decode(input_ids)
            
            # Generate text
            outputs = self.generator(
                truncated_prompt,
                max_new_tokens=200,  # Limit generation length
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Return the complete, unfiltered model output
            summary = outputs[0]['generated_text'].replace(truncated_prompt, "").strip()
            
            # print(summary)
            return summary
        
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return f"Error generating summary: {str(e)}"
    
    def __del__(self):
        """Clean up resources when done"""
        try:
            # Free up GPU memory
            if hasattr(self, 'model'):
                del self.model
            if hasattr(self, 'generator'):
                del self.generator
            torch.cuda.empty_cache()
        except:
            pass


class OllamaLLMProcessor:
    def __init__(self, model_name="llama2", use_long_prompt=True, host=None, port=None):
        """
        Initialize the LLM processor with Ollama local model.
        
        Args:
            model_name: Name of the Ollama model to use (e.g., "llama2", "mistral", etc.)
            use_long_prompt: Whether to use detailed prompt (True) or short prompt (False)
            host: Ollama host (defaults to environment variable OLLAMA_HOST or localhost)
            port: Ollama port (defaults to environment variable OLLAMA_PORT or 11434)
        """
        self.model_name = model_name
        self.use_long_prompt = use_long_prompt
        
        # Get host and port from environment variables or parameters
        self.host = host or os.getenv('OLLAMA_HOST', 'localhost')
        self.port = port or os.getenv('OLLAMA_PORT', '11434')
        
        # Clean up host if it includes protocol
        if '://' in self.host:
            self.host = self.host.split('://')[-1]
        
        # Remove port from host if included
        if ':' in self.host:
            self.host, port_from_host = self.host.split(':')
            if not port:  # Only use port from host if not explicitly specified
                self.port = port_from_host
                
        self.base_url = f"http://{self.host}:{self.port}"
        logger.info(f"Initializing Ollama model: {model_name} at {self.base_url}")
        
        # Test connection to Ollama server
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                logger.info("Successfully connected to Ollama server")
            else:
                raise ConnectionError(f"Could not connect to Ollama server. Status code: {response.status_code}")
        except Exception as e:
            logger.error(f"Error connecting to Ollama server at {self.base_url}: {str(e)}")
            raise

    def query_ollama(self, prompt):
        """Send a query to the Ollama API and get the response."""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False
                }
            )
            
            # Check if request was successful
            if response.status_code != 200:
                error_msg = f"Ollama API returned status code {response.status_code}"
                logger.error(error_msg)
                logger.error(f"Full response: {response.text}")
                return f"Error: {error_msg}"
            
            # Parse response and handle potential JSON decode errors
            try:
                response_json = response.json()
            except json.JSONDecodeError as e:
                error_msg = f"Failed to decode JSON response: {str(e)}\nRaw response: {response.text}"
                logger.error(error_msg)
                return f"Error: {error_msg}"
            
            # Check if response has the expected structure
            if not isinstance(response_json, dict):
                error_msg = f"Unexpected response format. Expected dict, got {type(response_json)}"
                logger.error(error_msg)
                return f"Error: {error_msg}"
            
            # Extract the response text - handle different possible response formats
            if "response" in response_json:
                return response_json["response"]
            elif "data" in response_json:
                return response_json["data"]
            elif "content" in response_json:
                return response_json["content"]
            else:
                error_msg = f"Response missing expected fields. Available fields: {list(response_json.keys())}"
                logger.error(error_msg)
                logger.error(f"Full response: {response_json}")
                return f"Error: {error_msg}"
                
        except requests.exceptions.ConnectionError as e:
            error_msg = f"Failed to connect to Ollama server at {self.base_url}: {str(e)}"
            logger.error(error_msg)
            return f"Error: {error_msg}"
        except Exception as e:
            error_msg = f"Unexpected error querying Ollama: {str(e)}"
            logger.error(error_msg)
            return f"Error: {error_msg}"

    def generate_summary(self, impression):
        """
        Generate a patient-friendly summary of the radiology impression using Ollama.
        """
        try:
            # Create the prompt based on length preference
            if self.use_long_prompt:
                prompt = f"""
                You are a medical communication specialist. Your task is to convert the following radiology impression into a clear, simple summary that a patient without medical background can understand.

                Focus on:
                1. Explaining medical terms in plain language
                2. Highlighting what these findings mean for the patient
                3. Avoiding unnecessarily alarming language
                4. Being honest but reassuring where appropriate

                You will be evaluated in terms of completeness, conciseness, accuracy, and existence of hallucinations.
                        
                Radiology Impression:
                {impression}
                
                Patient-Friendly Explanation:
                """
            else:
                prompt = f"""Create a patient-friendly summary of the following radiology impression:
                {impression}
                """
            
            # Generate text using Ollama
            logger.info(f"Generating summary for impression of length {len(impression)}")
            summary = self.query_ollama(prompt)
            
            logger.info(f"Generated summary of length {len(summary)}")
            # print(f"FULL MODEL OUTPUT:\n{summary}\n{'='*50}")
            
            return summary.strip()
        
        except Exception as e:
            logger.error(f"Error generating summary with Ollama: {str(e)}")
            return f"Error generating summary: {str(e)}"
    
    def __del__(self):
        """Clean up resources when done"""
        pass  # No specific cleanup needed for API client


def extract_impression(report_text):
    """
    Extract the impression section from a radiology report.
    Only extracts numbered sequences that start with 1. and continue sequentially.
    Ignores other numbered items that are not part of the main sequence.
    If no impression or numbered list is found, returns the whole text.
    """
    # print(f"\n=== Starting impression extraction ===")
    # print(f"Input text length: {len(report_text)} characters")
    
    # First try to find an explicit IMPRESSION section with numbered points
    # print("Looking for IMPRESSION section with numbered points...")
    try:
        # print(report_text)
        impression_section = re.search(r"IMPRESSION:\s*((?:\s*\d+\.\s+.*?)*?)(?=\n\s*(?:[A-Z]+:|$))", report_text, re.DOTALL | re.IGNORECASE, timeout=10)
    except re.TimeoutError:
        print("Regex search timed out, moving to next step")
        impression_section = None
    
    if impression_section:
        # print("Found IMPRESSION section, checking for valid content...")
        impression = impression_section.group(1).strip()
        if impression:
            # print(f"Impression content length: {len(impression)}")
            # Verify it contains a sequence starting with 1.
            if re.search(r"(?:^|\n)\s*1\.\s+", impression):
                # print("Found valid numbered sequence in IMPRESSION section")
                return impression
            else:
                print("No numbered sequence found in IMPRESSION section")
    else:
        print("No IMPRESSION section with numbered points found")
    
    # If no valid IMPRESSION section found, look for a sequence starting with 1. anywhere in the text
    # print("\nLooking for numbered sequence anywhere in text...")
    sequence_start = re.search(r"(?:^|\n)\s*1\.\s+.*?(?=(?:\n|$))", report_text, re.DOTALL)
    
    if sequence_start:
        # print("Found sequence starting with '1.'")
        start_pos = sequence_start.start()
        # print(f"Sequence starts at position: {start_pos}")
        # Now extract all consecutive numbered points from this position
        remaining_text = report_text[start_pos:]
        # print(f"Remaining text length: {len(remaining_text)}")
        numbered_points = []
        expected_number = 1
        max_iterations = 10  # Safety limit for maximum number of points
        
        # print("\nStarting numbered point extraction loop...")
        while expected_number <= max_iterations:
            # print(f"Looking for point #{expected_number}...")
            next_point = re.search(rf"(?:^|\n)\s*{expected_number}\.\s+.*?(?=(?:\n\s*\d+\.|$))", remaining_text, re.DOTALL)
            if not next_point:
                print(f"No point #{expected_number} found, breaking loop")
                break
            
            point_text = next_point.group().strip()
            # print(f"Found point #{expected_number}, length: {len(point_text)}")
            
            if not point_text:
                print(f"Point #{expected_number} is empty, breaking loop")
                break
                
            numbered_points.append(point_text)
            remaining_text = remaining_text[next_point.end():]
            # print(f"Remaining text length after point #{expected_number}: {len(remaining_text)}")
            expected_number += 1
            
            if not re.search(r"(?:^|\n)\s*\d+\.\s+", remaining_text):
                # print("No more numbered points found, breaking loop")
                break
        
        if numbered_points:
            # print(f"\nFound {len(numbered_points)} valid numbered points")
            return "\n".join(numbered_points)
        else:
            print("No valid numbered points collected")
    else:
        print("No sequence starting with '1.' found")
    
    # If still no match, look for any text after "IMPRESSION:" until the next section
    # print("\nLooking for plain IMPRESSION section...")
    impression_match = re.search(r"IMPRESSION:\s*(.*?)(?=\n\s*[A-Z]+:|$)", report_text, re.DOTALL | re.IGNORECASE, timeout=10)
    if impression_match:
        impression_text = impression_match.group(1).strip()
        if impression_text:
            # print("Found valid IMPRESSION section text")
            return impression_text
        else:
            print("Found empty IMPRESSION section")
    else:
        print("No plain IMPRESSION section found")
    
    # If no impression or numbered list found, return the whole text
    print(f"\nNo impression or numbered list found, returning whole text of length {len(report_text)}")
    return report_text.strip()


def process_batches(df, llm_processor, output_file, report_type="original", batch_size=10, max_workers=3, difficulty="hard"):
    """
    Summarize the first 100 rows of df in batches using multiple threads, writing the summary
    column(s) into df in place and checkpointing to <output_file>.partial after each batch.

    Args:
        df: Input dataframe
        llm_processor: LLM processor instance
        output_file: Path to output file
        report_type: Type of report ("original" or "attack")
        batch_size: Number of reports to process in each batch
        max_workers: Maximum number of concurrent threads
        difficulty: Difficulty of the adversarial column to summarize (attack reports only)
    """
    total_reports = len(df)
    all_results = []
    
    def process_report(row, idx, difficulty):
        """Process a single report based on report type"""
        if report_type == "attack":
            try:
                impression = row[f'adversarial_report_alter_terms_{difficulty}']
            except:
                impression = row[f'adversarial_report_missing_facts_{difficulty}']
            
            summary = llm_processor.generate_summary(impression)
            # Update DataFrame directly with correct index
            df.at[idx, f'patient_friendly_summary_adversarial_{difficulty}'] = summary
            
        else:  # original
            # if "Findings" is a column, use it to extract the impression
            # otherwise, use the "Impression" column
            if 'Findings' in row:
                impression = extract_impression(row['Findings'])
            else:
                impression = row['impressions'].strip()
            if impression and impression != "No impression found.":
                summary = llm_processor.generate_summary(impression)
                df.at[idx, 'patient_friendly_summary'] = summary
            else:
                df.at[idx, 'patient_friendly_summary'] = "No impression found to summarize."
            
        print(f"IMPRESSION:\n{impression}\n{'-'*50}")
        print(f"SUMMARY:\n{summary}\n{'='*50}")
    
    def process_batch(batch_df, difficulty):
        """Process a batch of reports"""
        for idx, row in batch_df.iterrows():
            logger.info(f"Processing report {idx}")
            process_report(row, idx, difficulty)
    
    # Process in batches using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        
        # Summaries are generated for the FIRST 100 rows of the input CSV, in file order.
        # These 100 reports per dataset are the evaluation subset used in the paper.
        for i in range(0, 100, batch_size):
            end_idx = min(i + batch_size, total_reports)
            batch_df = df.iloc[i:end_idx]
            futures.append(
                executor.submit(process_batch, batch_df, difficulty)
            )
        
        # Process results as they complete
        for future in tqdm(futures, desc="Processing batches"):
            future.result()
            
            # Save intermediate results with all columns
            df.to_csv(f"{output_file}.partial", index=False)
            logger.info(f"Saved intermediate results")
    
    # Save final results with all columns
    df.to_csv(output_file, index=False)
    logger.info(f"Processing complete. Results saved to {output_file}")
    
    return df


def main(input_file, output_file, model_name="llama2", batch_size=1, max_workers=3,
         model_type="ollama", api_key=None, use_long_prompt=True, report_type="attack",
         ollama_host=None, ollama_port=None, difficulty="hard",vllm_base_url=None,
         azure_endpoint=None, azure_deployment=None, azure_api_version="2025-04-01-preview"):
    """
    Main function to process the CSV file of radiology reports.

    Args:
        input_file: Path to input CSV file containing radiology reports
        output_file: Path to output CSV file to save results
        model_name: Name of the model to use
        batch_size: Number of reports to process in each batch
        max_workers: Maximum number of concurrent threads
        model_type: "openai", "azure", "claude", "ollama", "vllm", "nvidia" or "local"
        api_key: API key for the selected backend (otherwise read from its environment variable)
        use_long_prompt: Whether to use detailed prompt (True) or short prompt (False)
        report_type: Type of report ("original" or "attack")
        ollama_host: Host for Ollama server (optional)
        ollama_port: Port for Ollama server (optional)
        difficulty: Difficulty level for attack reports ("easy", "medium", "hard")
        vllm_base_url: Base URL for vLLM OpenAI-compatible server (e.g., http://localhost:8000/v1)
        azure_endpoint: Azure OpenAI endpoint (otherwise AZURE_OPENAI_ENDPOINT)
        azure_deployment: Azure deployment name (defaults to model_name)
        azure_api_version: Azure OpenAI API version
    """
    try:
        logger.info(f"========== Model: {model_name} | Type: {model_type} ==========")
        # Load CSV file
        logger.info(f"Loading CSV file: {input_file}")
        df = pd.read_csv(input_file)
        # keep rows where Split is test or val
        df = df[df['Split'].isin(['test', 'val'])]
        total_reports = len(df)
        logger.info(f"Found {total_reports} reports to process")
        
        cost_tracker = CostTracker()
        
        # Validate report type and required columns
        if report_type == "attack" and f'adversarial_report_alter_terms_{difficulty}' not in df.columns and f'adversarial_report_missing_facts_{difficulty}' not in df.columns:
            raise ValueError("Attack reports require 'adversarial_report' column")
        # either Findings or impressions columns should be present, we don't need both
        elif report_type == "original":
            if 'Findings' not in df.columns and 'impressions' not in df.columns: 
                raise ValueError("Original reports require 'Findings' or 'impressions' columns")
        
        # Initialize the LLM processor based on model type
        if model_type == "nvidia":
            llm_processor = NVIDIALLMProcessor(model_name=model_name, api_key=api_key, use_long_prompt=use_long_prompt)
        elif model_type == "local":
            llm_processor = OpenSourceLLMProcessor(model_name=model_name, use_long_prompt=use_long_prompt)
        elif model_type == "ollama":
            llm_processor = OllamaLLMProcessor(
                model_name=model_name,
                use_long_prompt=use_long_prompt,
                host=ollama_host,
                port=ollama_port
            )
        elif model_type == "vllm":
            llm_processor = VLLMLLMProcessor(
                model_name=model_name,
                base_url=vllm_base_url,
                api_key=api_key,
                use_long_prompt=use_long_prompt
            )
        elif model_type == "openai":
            llm_processor = OpenAIProprietaryProcessor(
                model_name=model_name,
                api_key=api_key,
                use_long_prompt=use_long_prompt,
                cost_tracker=cost_tracker
            )
        elif model_type == "claude":
            llm_processor = ClaudeLLMProcessor(
                model_name=model_name,
                api_key=api_key,
                use_long_prompt=use_long_prompt,
                cost_tracker=cost_tracker
            )
        elif model_type == "azure":
            logger.info(f"Using Azure OpenAI processor for model {model_name}")
            llm_processor = AzureOpenAIProcessor(
                model_name=model_name,
                deployment=(azure_deployment or model_name),
                endpoint=azure_endpoint,
                api_version=azure_api_version,
                api_key=api_key,
                use_long_prompt=use_long_prompt,
                cost_tracker=cost_tracker
            )

        else:
            raise ValueError(f"Invalid model type: {model_type}")
        
        # Process in batches
        updated_df = process_batches(df, llm_processor, output_file, report_type, batch_size, max_workers, difficulty)
        
        # Save the updated dataframe with new summary columns
        updated_df.to_csv(output_file, index=False)
        logger.info(f"Processing complete. Results saved to {output_file}")
        
        if cost_tracker:
            cost_tracker.summarize()

    except Exception as e:
        logger.error(f"Error in main processing: {str(e)}")
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Path to input CSV file (output of step 1)")
    parser.add_argument("--output", required=True, help="Path to output CSV file")
    parser.add_argument("--model", default="llama2", help="Model name as known to the backend (see README table)")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of reports to process in each batch")
    parser.add_argument("--max-workers", type=int, default=2, help="Maximum number of concurrent threads")
    parser.add_argument(
    "--model-type",
    choices=["nvidia", "local", "ollama", "vllm", "openai", "claude", "azure"],
    default="ollama",
    help="Backend to use (nvidia, local, ollama, vllm, openai, claude, azure). Default: ollama"
    )
    parser.add_argument("--azure-endpoint", help="Azure OpenAI endpoint, e.g. https://<res>.cognitiveservices.azure.com/")
    parser.add_argument("--azure-api-version", default="2025-04-01-preview", help="Azure OpenAI API version")
    parser.add_argument("--azure-deployment", help="Azure OpenAI deployment name (defaults to --model)")

    parser.add_argument("--api-key", help="API key for the selected backend (default: read from that "
                                          "backend's environment variable, see README)")
    parser.add_argument("--short-prompt", default=False, action="store_true",
                        help="Use the concise prompt instead of the detailed (default) prompt")
    parser.add_argument("--report-type", choices=["original", "attack"], default="attack",
                      help="Summarize the original impression or an adversarial one. Default: attack")
    parser.add_argument("--ollama-host", help="Host for Ollama server (default: OLLAMA_HOST or localhost)")
    parser.add_argument("--ollama-port", help="Port for Ollama server (default: OLLAMA_PORT or 11434)")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="hard",
                      help="Difficulty level for attack reports (easy, medium, hard). Default: hard")
    parser.add_argument("--vllm-base-url", help="Base URL for vLLM OpenAI server, e.g., http://localhost:8000/v1")

    args = parser.parse_args()
    
    main(
        input_file=args.input,
        output_file=args.output,
        model_name=args.model,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        model_type=args.model_type,
        api_key=args.api_key,
        use_long_prompt=not args.short_prompt,
        report_type=args.report_type,
        ollama_host=args.ollama_host,
        ollama_port=args.ollama_port,
        difficulty=args.difficulty,
        vllm_base_url=args.vllm_base_url,
        azure_endpoint=args.azure_endpoint,
        azure_deployment=args.azure_deployment,
        azure_api_version=args.azure_api_version,
    )