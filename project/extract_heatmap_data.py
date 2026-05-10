import json
import pandas as pd
import numpy as np
from collections import defaultdict

# Utility to load JSON data
def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

# Group entities by document ID and normalize text
def get_entities_by_doc(data):
    grouped = defaultdict(list)
    for doc in data:
        doc_id = doc.get("doc_id")
        entities = doc.get("predicted_entities", doc.get("true_entities", []))
        for ent in entities:
            grouped[doc_id].append({
                "start": ent.get("start"),
                "end": ent.get("end"),
                "text": ent.get("text").lower().strip()
            })
    return grouped

# Main analysis function for 3x3 performance grid
def extract_heatmap_grid_3x3_full():
    df_tokens = pd.read_csv('project/token_analysis.csv')
    token_lookup = dict(zip(df_tokens['entity_text'].str.lower().str.strip(), df_tokens['biobert_token_count']))

    # Merge Easy and Hard datasets for full evaluation
    gold_combined = load_json('project/test_easy.json') + load_json('project/test_hard.json')
    pred_combined = load_json('project/biobert_predictions_easy.json') + load_json('project/biobert_predictions_hard.json')

    gold_grouped = get_entities_by_doc(gold_combined)
    pred_grouped = get_entities_by_doc(pred_combined)

    # Define dimensions for the 3x3 grid
    char_buckets = ["1-5 chars", "6-10 chars", "11+ chars"]
    token_buckets = ["1 Token", "2 Tokens", "3+ Tokens"]
    grid = {t: {c: {"TP": 0, "FP": 0, "FN": 0} for c in char_buckets} for t in token_buckets}

    # Categorization logic for tokens and character lengths
    def get_token_bucket(text):
        count = token_lookup.get(text, 1)
        if count == 1: return "1 Token"
        elif count == 2: return "2 Tokens"
        else: return "3+ Tokens"

    def get_char_bucket(text):
        length = len(text)
        if length <= 5: return "1-5 chars"
        elif length <= 10: return "6-10 chars"
        else: return "11+ chars"

    # Exact match span comparison and bucket counting
    all_doc_ids = set(gold_grouped.keys()).union(set(pred_grouped.keys()))

    for doc_id in all_doc_ids:
        gold_ents = {(e["start"], e["end"], e["text"]) for e in gold_grouped[doc_id]}
        pred_ents = {(e["start"], e["end"], e["text"]) for e in pred_grouped[doc_id]}

        gold_spans = gold_ents.intersection(pred_ents)
        for span in gold_spans:
            grid[get_token_bucket(span[2])][get_char_bucket(span[2])]["TP"] += 1

        for span in pred_ents - gold_ents:
            grid[get_token_bucket(span[2])][get_char_bucket(span[2])]["FP"] += 1

        for span in gold_ents - pred_ents:
            grid[get_token_bucket(span[2])][get_char_bucket(span[2])]["FN"] += 1

    # Formatting and printing results as a dictionary
    print("\n=== COPY AND PASTE THIS INTO YOUR HEATMAP SCRIPT ===\n")
    print("    data = {")
    for c in char_buckets:
        scores = []
        for t in token_buckets:
            counts = grid[t][c]
            tp, fp, fn = counts["TP"], counts["FP"], counts["FN"]
            
            if (tp + fp + fn) < 5:
                scores.append("np.nan")
            else:
                p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
                scores.append(f"{f1:.4f}")
        
        row_str = ", ".join([f"{s:>8}" for s in scores])
        print(f'        "{c}": [{row_str}],')
    print("    }")

# Execute script
if __name__ == '__main__':
    extract_heatmap_grid_3x3_full()