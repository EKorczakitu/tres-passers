# tres-passers
# Medical NER Analysis: BioBERT vs. Gemini

This repository contains a comparative analysis of **BioBERT** (a fine-tuned encoder model) and **Gemini** (a zero-shot generative LLM) for Named Entity Recognition (NER) in the medical domain. The research specifically focuses on how sub-word token fragmentation impacts model performance.

## 1. Setup and Installation

### Prerequisites
- Python 3.9 or higher
- A Google Gemini API Key (if running the LLM pipeline)

### Installation
Install all required libraries using the provided requirements file:
```bash
pip install -r requirements.txt


Follow these phases in numerical order to reproduce the analysis.

Phase 1: Data Stratification
File: 1_data_stratification.ipynb

What it does: Downloads the BC5CDR dataset from Hugging Face. It reconstructs document strings and calculates the token count for every medical entity using the BioBERT tokenizer.

Logic: It splits the data into two buckets:

Easy: Documents where all entities are 1-2 tokens long.

Hard: Documents where at least one entity is fragmented into 3+ tokens.

Output: test_easy.json, test_hard.json, and token_analysis.csv.

Phase 2: BioBERT Fine-tuning & Inference
File: 2_biobert_finetuning.ipynb

What it does: Fine-tunes the dmis-lab/biobert-base-cased-v1.1 model on the BC5CDR training set for 3 epochs.

Inference: The model then processes the test_easy.json and test_hard.json files generated in Phase 1. It converts BIO tags back into character-level spans to match the original text.

Output: biobert_predictions_easy.json and biobert_predictions_hard.json.

Phase 3: Gemini Zero-Shot Pipeline
File: 3_gemini_zero_shot.ipynb

What it does: 1. Extraction: Sends the text from the test sets to the Gemini API asynchronously (using asyncio) to extract entities in JSON format.
2. Mapping: Since Gemini provides strings, this script uses Regex boundary matching to find the exact character offsets (start and end) within the original document text.

Output: gemini_predictions_easy.json and gemini_predictions_hard.json.

Phase 4: Final Evaluation & Visualizations
Once all prediction files are ready, run the following standalone scripts:

rq2analysis.py: Calculates the core performance metrics (Strict vs. Relaxed F1) for all models across the Easy and Hard datasets.

analyze_fragmentation.py: Specifically analyzes how F1-score drops as token count increases.

Result: Generates fragmentation_impact.png.

extract_fp.py: Identifies 1-token "hallucinations"—cases where the model predicts an entity that isn't in the gold standard.

Result: Generates gemini_1_token_errors.csv.

plots.ipynb: A notebook for generating final heatmaps and correlation plots used in the research discussion.