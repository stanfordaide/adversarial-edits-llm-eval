r"""
Step 1 of the pipeline: generate adversarial radiology impressions.

For every test-split report in the input CSV, the impression is extracted (Merlin, MIMIC-IV:
from the `Findings` column with `extract_impression`; INSPECT: the `impressions` column is used
as is) and perturbed at three difficulty levels (easy / medium / hard) with one of two strategies:

  alter_terms    (term substitution) replace terms found in the term dictionary with one of
                 their listed alternatives. easy = 1 term in 1 category, medium = up to 2 terms
                 in each of 2 categories, hard = up to 4 terms in each of 4 categories; the
                 least-used terms are preferred so substitutions are spread over the dictionary.
                 Every measurement ("1.5 cm", "8 x 8 x 5 mm") is then replaced by a random
                 value and unit with 50% probability.
  missing_facts  (impression removal) remove the numbered impression points that contain the
                 most medical terms (from data/medical_terms/): easy = 1, medium = up to 2,
                 hard = up to 3, never more than half of the points. Impressions with a single
                 point fall back to removing sentences, and single sentences to removing one
                 random medical word.

Input:  a CSV with a `Split` column (only rows with Split == "test" are kept) and the report
        text in `Findings` (Merlin, MIMIC-IV) or `impressions` (INSPECT).
Output: <input>_with_adversarial_<strategy>.csv next to the input, i.e. the test rows plus
        adversarial_report_<strategy>_<difficulty>   the perturbed impression
        ground_truth_<strategy>_<difficulty>         the list of edits that were made
        for difficulty in easy / medium / hard.

The paper's adversarial reports were generated without a random seed; pass --seed to make
new runs repeatable. The default dictionary (v1_as_used) is the one used in the paper.

Usage:
  python pipeline/adversarial_report_generation.py --dataset merlin --strategy alter_terms \
      --input_csv <merlin_reports.csv>
"""
import random
import json
import argparse
from typing import List, Tuple, Set, Dict
from collections import Counter
import re
import pandas as pd
from tqdm import tqdm
from generate_patient_friendly import extract_impression
import spacy


class AdversarialReportGenerator:
    """Applies term-substitution or impression-removal edits to radiology impressions."""

    def __init__(self, medical_terms_config: str, term_alternatives_config: str, dataset: str):
        """
        Args:
            medical_terms_config: JSON file {"medical_terms": [...]} used to rank impression
                points / sentences for impression removal.
            term_alternatives_config: term dictionary JSON
                {"term_alternatives": {term: {"alternatives": [...], "category": ...}}}.
            dataset: dataset name (kept for reference; not used by the edits themselves).
        """
        self.nlp = spacy.load("en_core_sci_md")
        
        # Load medical terms from config file
        with open(medical_terms_config, 'r') as file:
            config = json.load(file)
            self.medical_terms = set(config['medical_terms'])
        
        # Load term alternatives from config file
        with open(term_alternatives_config, 'r') as file:
            config = json.load(file)
            self.term_alternatives = config['term_alternatives']

        # Initialize counter to track term usage
        self.term_usage = Counter()

        # Categorize terms based on their category in term_alternatives
        self.categorized_terms = {
            'anatomy': set(),
            'pathology': set(),
            'finding': set(),
            'study_types': set(),
            'descriptors': set(),
            'location': set(),
            'confidence_terms': set()
        }

        # Populate categorized terms
        for term, info in self.term_alternatives.items():
            category = info['category']
            if category in self.categorized_terms:
                self.categorized_terms[category].add(term)

    
    def alter_measurements(self, tokens: List[Tuple[str, str]], alterations: List[Tuple[str, str]]) -> None:
        """
        Alter measurements in the text by changing numbers and units with a 50% chance per occurrence.
        Each altered measurement is appended to `alterations`; `tokens` is replaced in place.
        Handles:
        - Single: "1.1 cm", "9.8 mm", "5.8-mm", "8mm"
        - Dimensions: "8 x 8 x 5 cm", "1.2 × 3.4 mm"
        """
        # Reconstruct the full text once; current tokenization splits numbers/units into separate tokens.
        original_text = ''.join(t[1] for t in tokens)

        # number(s) [x or × number(s) ...] optional spaces/hyphen then unit (cm|mm)
        pattern = re.compile(
            r'\b('
            r'\d+(?:\.\d+)?(?:\s*[x×]\s*\d+(?:\.\d+)?)*'   # 1.2  or  1.2 x 3.4  or  8×8×5
            r')(\s*-?\s*)'                                  # optional spaces/hyphen between value(s) and unit
            r'(cm|mm)\b',                                   # unit
            re.IGNORECASE
        )

        def repl(m: re.Match) -> str:
            original = m.group(0)
            nums_block = m.group(1)         # e.g., "8 x 8 x 5" or "6" or "5.8"
            sep = m.group(2)                # e.g., " ", "", " - "
            unit = m.group(3)               # "cm" or "mm"

            # 50% chance to change; otherwise leave as-is
            if random.random() >= 0.5:
                return original

            # Split by x/× to keep the count of dimensions
            parts = re.split(r'\s*[x×]\s*', nums_block)
            # Generate new numbers (keep count, 1 decimal)
            new_parts = [f"{round(random.uniform(1.0, 10.0), 1):.1f}" for _ in parts]

            # Preserve the multiplication symbol style if present
            if '×' in nums_block:
                joiner = ' × '
            elif re.search(r'\s*x\s*', nums_block, flags=re.IGNORECASE):
                joiner = ' x '
            else:
                # no multiplier (single number)
                joiner = ''

            new_nums_block = joiner.join(new_parts) if joiner else new_parts[0]

            # Randomly choose a unit
            new_unit = random.choice(['mm', 'cm'])

            new_text = f"{new_nums_block}{sep}{new_unit}"

            # Track alteration for ground truth
            alterations.append((original, new_text))
            return new_text

        new_text = pattern.sub(repl, original_text)

        # Replace the entire token stream with the updated text (safe: later code just joins tokens)
        tokens[:] = [("WORD", new_text)]


    def alter_terms(self, text: str, difficulty: str) -> Tuple[str, str]:
        """
        Term substitution. Replaces dictionary terms with a random alternative from the same
        dictionary entry (see the module docstring for the per-difficulty counts), then alters
        measurements. Returns (adversarial_text, ground_truth), where ground_truth lists every
        edit as "Altered terms:\\n1. <original> -> <replacement>\\n...".
        """
        # First, split the text into tokens while preserving their positions
        tokens = []
        current_word = ""
        current_punct = ""
        
        for char in text:
            if char.isalnum() or char == ' ' or char == '-':  # Added hyphen for measurements
                if current_punct:
                    if current_punct.strip():  # If it's not just whitespace
                        tokens.append(("PUNCT", current_punct))
                    current_punct = ""
                if char == ' ':
                    if current_word:
                        tokens.append(("WORD", current_word))
                        current_word = ""
                    tokens.append(("SPACE", " "))
                else:
                    current_word += char
            else:
                if current_word:
                    tokens.append(("WORD", current_word))
                    current_word = ""
                current_punct += char
        
        # Add any remaining tokens
        if current_word:
            tokens.append(("WORD", current_word))
        if current_punct:
            tokens.append(("PUNCT", current_punct))

        # Extract just the words for processing
        words = [token[1] for token in tokens if token[0] == "WORD"]
        word_indices = [i for i, token in enumerate(tokens) if token[0] == "WORD"]
        
        alterations = []
        altered_indices = set()
        
        def normalize_term(term: str) -> str:
            """Normalize a term by removing trailing 's' and handling slashes."""
            term = term.lower()
            # Remove trailing 's' if present
            if term.endswith('s') and not term.endswith('is'):
                term = term[:-1]
            # Handle terms with slashes by trying both parts
            if '/' in term:
                parts = term.split('/')
                return parts[0]  # Use the first part
            return term
        
        def find_term_in_category(word: str, category: str) -> str:
            """Find if a word matches any term in the category, handling plurals and slashes."""
            normalized = normalize_term(word)
            # Check the normalized version against category terms
            for cat_term in self.categorized_terms[category]:
                cat_normalized = normalize_term(cat_term)
                if normalized == cat_normalized:
                    return cat_term  # Return the original dictionary term
            return None
        
        def alter_specific_category(category: str, num: int, verbose: bool = False):
            nonlocal words, alterations, altered_indices
            # Find terms in the text that belong to the specified category
            category_terms = []
            for i, word in enumerate(words):
                if i not in altered_indices:
                    # Try to find a matching term in the category
                    matched_term = find_term_in_category(word, category)
                    if matched_term:
                        category_terms.append((i, word, matched_term))
            
            if verbose:
                print(f"Category terms: {category_terms}")
                
            # Sort terms by usage (least used first)
            category_terms.sort(key=lambda x: self.term_usage[x[2].lower()])
            
            # Alter terms up to the specified number
            for _ in range(min(num, len(category_terms))):
                if category_terms:
                    index, original_word, dict_term = category_terms.pop(0)  # Take the least used term
                    replacement = random.choice(self.term_alternatives[dict_term.lower()]['alternatives'])
                    words[index] = replacement
                    alterations.append((original_word, replacement))
                    altered_indices.add(index)
                    self.term_usage[dict_term.lower()] += 1
                    self.term_usage[replacement.lower()] += 1
            
            return len(alterations)  # Return the number of alterations made

        # Shuffle categories for randomization
        categories = list(self.categorized_terms.keys())
        random.shuffle(categories)

        # Set difficulty parameters
        if difficulty == 'easy':
            terms_per_category = 1
            categories_to_alter = 1
        elif difficulty == 'medium':
            terms_per_category = 2
            categories_to_alter = 2
        else:  # hard
            terms_per_category = 4
            categories_to_alter = 4

        # Alter terms based on difficulty
        total_alterations = 0
        for category in categories:
            if total_alterations >= categories_to_alter * terms_per_category:
                break
            alterations_made = alter_specific_category(category, terms_per_category)
            total_alterations += alterations_made

        # Try other categories, but avoid infinite loop
        attempts = 0
        max_attempts = len(categories)*10  # Limit attempts to 10 passes over the categories
        while total_alterations < categories_to_alter and attempts < max_attempts:
            attempts += 1
            made_alteration = False
            for category in categories:
                if total_alterations >= categories_to_alter * terms_per_category:
                    break
                alterations_made = alter_specific_category(category, 1)
                if alterations_made > 0:
                    made_alteration = True
                    total_alterations += alterations_made
            if not made_alteration:  # If we couldn't make any alterations, stop trying
                alterations_made = alter_specific_category(category, 1, verbose=True)
                print(f"Category terms: {terms_per_category}")
                break

        # Handle two-word phrases if still not enough alterations
        if total_alterations < categories_to_alter:
            # Modified to handle phrases of any length
            i = 1
            while i < len(words):
                if i not in altered_indices and i-1 not in altered_indices:
                    # Try different length phrases starting at position i-1
                    for length in range(2, min(4, len(words) - i + 2)):  # Try phrases up to 3 words
                        phrase = ' '.join(words[i-1:i-1+length])
                        # Try both the original phrase and normalized version
                        phrase_lower = phrase.lower()
                        normalized_phrase = normalize_term(phrase)
                        if phrase_lower in self.term_alternatives or normalized_phrase in self.term_alternatives:
                            original = phrase
                            dict_term = phrase_lower if phrase_lower in self.term_alternatives else normalized_phrase
                            replacement = random.choice(self.term_alternatives[dict_term]['alternatives'])
                            replacement_words = replacement.split()
                            
                            # Replace the original words with replacement words
                            words[i-1:i-1+length] = replacement_words
                            alterations.append((original, replacement))
                            # Update altered indices for all affected positions
                            altered_indices.update(range(i-1, i-1+len(replacement_words)))
                            
                            self.term_usage[dict_term] += 1
                            self.term_usage[replacement.lower()] += 1
                            total_alterations += 1
                            
                            # Skip the replaced words in the next iteration
                            i += len(replacement_words)
                            break  # Only alter one phrase
                    else:
                        i += 1  # Move to next word if no phrase was found
                else:
                    i += 1  # Skip already altered words
                
                if total_alterations >= categories_to_alter:
                    break

        # Rebuild the token structure with the altered words
        # Since phrase replacements may have changed the number of words,
        # we need to rebuild the entire token structure
        new_tokens = []
        word_idx = 0
        
        for token in tokens:
            if token[0] == "WORD":
                if word_idx < len(words):
                    new_tokens.append(("WORD", words[word_idx]))
                    word_idx += 1
                else:
                    # If we have more WORD tokens than words, skip them
                    continue
            else:
                new_tokens.append(token)
        
        tokens = new_tokens

        # After handling term alterations, process measurements
        self.alter_measurements(tokens, alterations)

        # Reconstruct the text with original punctuation
        adversarial_report = ''.join(token[1] for token in tokens)
        ground_truth = "Altered terms:\n" + '\n'.join(f"{i+1}. {original} -> {replacement}" for i, (original, replacement) in enumerate(alterations))

        return adversarial_report, ground_truth

    def count_medical_terms(self, sentence: str) -> int:
        """Count the whitespace-separated words of `sentence` that are in the medical-term list."""
        words = sentence.lower().split()
        return sum(1 for word in words if word in self.medical_terms)

    def split_into_sentences(self, text: str) -> List[str]:
        # Split text into sentences
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def missing_facts(self, text: str, difficulty: str) -> Tuple[str, str]:
        """
        Impression removal. Removes the numbered impression points (or, for single-point
        impressions, the sentences) with the most medical terms; for a single sentence, removes
        one random medical word. Returns (adversarial_text, ground_truth), where ground_truth
        lists what was removed.
        """
        # Split text into numbered impressions, looking specifically for impression format
        # This pattern looks for: start of line or whitespace, followed by number, dot, whitespace
        impression_points = re.split(r'(?:^|\s)(?=\d+\.\s)', text)
        impression_points = [p.strip() for p in impression_points if p.strip()]
        
        # Further filter to ensure we only get actual numbered impressions
        # This removes any split that happened due to measurements like "7mm" or "3 cm"
        valid_impressions = []
        for point in impression_points:
            # Check if the point starts with a number followed by a dot and space
            if re.match(r'^\d+\.\s+\w', point):
                valid_impressions.append(point)
            elif valid_impressions:  # If it doesn't match but we have previous valid impressions
                # It might be part of the previous impression that got split wrongly
                valid_impressions[-1] = valid_impressions[-1] + " " + point
        
        impression_points = valid_impressions
        print(f"Valid impression points: {impression_points}")
        
        # If no numbered impressions were found, but we have text,
        # treat the whole text as a single impression.
        if not impression_points and text.strip():
            impression_points = [text.strip()]
        
        # If there's only one impression point, handle it differently
        if len(impression_points) == 1:
            impression_text = re.sub(r'^\d+\.\s*', '', impression_points[0])

            # 1. Split into sentences using spaCy
            doc = self.nlp(impression_text)
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

            # 2. Check if we should use sentence removal or fall back to word removal
            if len(sentences) > 1:
                # --- This is the sentence removal logic from before ---
                sentence_scores = sorted(
                    [(sent, self.count_medical_terms(sent)) for sent in sentences],
                    key=lambda x: x[1],
                    reverse=True
                )
                
                num_to_remove = {
                    'easy': 1,
                    'medium': max(1, len(sentences) // 3),
                    'hard': max(1, len(sentences) // 2)
                }[difficulty]
                
                num_to_remove = min(num_to_remove, len(sentences) - 1)
                
                sentences_to_remove = {sent for sent, score in sentence_scores[:num_to_remove]}
                
                new_sentences = [sent for sent in sentences if sent not in sentences_to_remove]
                removed_sentences = [sent for sent in sentences if sent in sentences_to_remove]
                
                adversarial_report = ' '.join(new_sentences)
                ground_truth = "Removed sentences:\n" + '\n'.join(f"- {s}" for s in removed_sentences)
            
            else:
                # Fallback to removing a single significant word
                words = impression_text.split()
                
                # Find all medical terms and their original indices in the sentence
                medical_words_indices = [
                    i for i, word in enumerate(words)
                    if self.count_medical_terms(word.strip('.,')) > 0
                ]

                # If no medical terms are found, make no changes
                if not medical_words_indices:
                    return text, "No changes made: no medical terms found in the single sentence."

                # Select one random medical word to remove
                index_to_remove = random.choice(medical_words_indices)
                
                removed_word = words[index_to_remove]
                # Reconstruct the sentence without the removed word
                new_words = words[:index_to_remove] + words[index_to_remove + 1:]
                
                adversarial_report = ' '.join(new_words)
                ground_truth = f"Removed word: {removed_word}"
        else:
            # Score impressions based on medical term count
            impression_scores = [(p, self.count_medical_terms(p)) for p in impression_points]
            impression_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Determine number of impressions to remove based on difficulty
            num_to_remove = {
                'easy': 1,
                'medium': min(2, len(impression_points) // 2),
                'hard': min(3, len(impression_points) // 2)
            }[difficulty]
            
            num_to_remove = min(num_to_remove, len(impression_points) - 1)
            
            # Select impressions to remove (just the impressions, not the scores)
            impressions_to_remove = [imp for imp, _ in impression_scores[:num_to_remove]]
            
            # Create new report with impressions removed
            new_impressions = []
            removed_impressions = []
            for impression in impression_points:
                if impression in impressions_to_remove:
                    removed_impressions.append(impression)  # Keep original numbering
                else:
                    new_impressions.append(impression)  # Keep original numbering
            
            adversarial_report = ' '.join(new_impressions)
            ground_truth = f"Removed impressions:\n" + '\n'.join(removed_impressions)
        
        return adversarial_report, ground_truth

    def generate_adversarial_report(self, original_report: str, difficulty: str, strategy: str) -> Tuple[str, str]:
        """Dispatch to alter_terms / missing_facts. Returns (adversarial_text, ground_truth)."""
        if strategy == 'missing_facts':
            return self.missing_facts(original_report, difficulty)
        elif strategy == 'alter_terms':
            return self.alter_terms(original_report, difficulty)
        else:
            raise ValueError(f"Unknown or unimplemented strategy: {strategy}")

    def process_reports_csv(self, strategy: str, csv_path: str, dataset: str):
        """
        Process reports from a CSV file, generate adversarial reports for all difficulties.

        Args:
            strategy: 'alter_terms' or 'missing_facts'
            csv_path: Path to the CSV file containing reports (needs a 'Split' column)
            dataset: 'merlin', 'inspect' or 'mimic'. INSPECT reports are read from the
                'impressions' column as is; the others from 'Findings' via extract_impression.

        Writes <csv_path without .csv>_with_adversarial_<strategy>.csv.
        """
        # Read the CSV file
        df = pd.read_csv(csv_path)
        df = df[df['Split'] == 'test']
        if dataset == 'inspect':
            df['Findings'] = df['impressions']
        # Process each difficulty level
        for difficulty in ['easy', 'medium', 'hard']:
            # Create new columns for this difficulty level
            df[f'adversarial_report_{strategy}_{difficulty}'] = ''
            df[f'ground_truth_{strategy}_{difficulty}'] = ''
            
            # Process each row
            for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {strategy} {difficulty}"):
                findings = row['Findings']
                if dataset == 'inspect':
                    impression = row['Findings']
                else:
                    impression = extract_impression(findings)
                if isinstance(impression, str):
                    # Generate adversarial report and ground truth
                    adversarial_report, ground_truth = self.generate_adversarial_report(
                        impression,
                        difficulty,
                        strategy
                    )
                    
                    # Save results back to DataFrame
                    df.at[idx, f'adversarial_report_{strategy}_{difficulty}'] = adversarial_report
                    df.at[idx, f'ground_truth_{strategy}_{difficulty}'] = ground_truth
        
        # Save the updated DataFrame back to CSV
        output_path = csv_path.replace('.csv', f'_with_adversarial_{strategy}.csv')
        df.to_csv(output_path, index=False)
        print(f"Saved results to {output_path}")

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strategy", choices=['missing_facts', 'alter_terms'], default='missing_facts',
                        help="Perturbation strategy: alter_terms (term substitution) or "
                             "missing_facts (impression removal). Default: missing_facts")
    parser.add_argument("--input_csv", required=True,
                        help="CSV with the reports (needs a 'Split' column; only rows with Split == 'test' are used). "
                             "Output is written next to it as <input>_with_adversarial_<strategy>.csv")
    parser.add_argument("--medical_terms_config", default=None,
                        help="JSON file containing extracted medical terms "
                             "(default: data/medical_terms/medical_terms_<dataset>.json)")
    parser.add_argument("--term_alternatives_config", default="dictionary/adversarial_term_dictionary_v1_as_used.json",
                        help="Term dictionary JSON with the alternatives for each term "
                             "(v1_as_used reproduces the paper; v2_cleaned is recommended for new work)")
    parser.add_argument("--dataset", choices=['mimic', 'merlin', 'inspect'], default='merlin',
                        help="Dataset of the input CSV; selects the report column and the default "
                             "medical-term list. Default: merlin")
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional random seed. The paper's adversarial reports were generated without a seed.")
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
    if args.medical_terms_config is None:
        args.medical_terms_config = {
            "inspect": "data/medical_terms/medical_terms_inspect.json",
            "merlin": "data/medical_terms/medical_terms_merlin.json",
            "mimic": "data/medical_terms/medical_terms_mimic_mri.json",
        }[args.dataset]
    
    generator = AdversarialReportGenerator(medical_terms_config=args.medical_terms_config,
                                         term_alternatives_config=args.term_alternatives_config, dataset=args.dataset)
    generator.process_reports_csv(strategy=args.strategy, csv_path=args.input_csv, dataset=args.dataset)

if __name__ == "__main__":
    main()