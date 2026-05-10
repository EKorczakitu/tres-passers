import json
import pandas as pd
from collections import defaultdict

# Data loading helper
def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

# Entity grouping by document ID
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

# Main analysis logic for Strict and Relaxed matching
def analyze_fragmentation_final(gold_file, pred_file, token_csv, model_name):
    df_tokens = pd.read_csv(token_csv)
    token_lookup = dict(zip(df_tokens['entity_text'].str.lower().str.strip(), df_tokens['biobert_token_count']))

    gold_grouped = get_entities_by_doc(load_json(gold_file))
    pred_grouped = get_entities_by_doc(load_json(pred_file))

    buckets = {
        "1 Token":   {"Strict TP": 0, "Strict FP": 0, "Strict FN": 0, "Relaxed TP": 0, "Relaxed FP": 0, "Relaxed FN": 0},
        "2 Tokens":  {"Strict TP": 0, "Strict FP": 0, "Strict FN": 0, "Relaxed TP": 0, "Relaxed FP": 0, "Relaxed FN": 0},
        "3+ Tokens": {"Strict TP": 0, "Strict FP": 0, "Strict FN": 0, "Relaxed TP": 0, "Relaxed FP": 0, "Relaxed FN": 0}
    }

    # Token bucket categorization logic
    def get_bucket_name(text):
        if text in token_lookup:
            count = token_lookup[text]
        else:
            count = max(1, len(text) // 4) 
        
        if count == 1: return "1 Token"
        elif count == 2: return "2 Tokens"
        else: return "3+ Tokens"

    all_doc_ids = set(gold_grouped.keys()).union(set(pred_grouped.keys()))

    for doc_id in all_doc_ids:
        golds = gold_grouped[doc_id]
        preds = pred_grouped[doc_id]

        # Strict Matching evaluation
        gold_spans = {(e["start"], e["end"], e["text"]) for e in golds}
        pred_spans = {(e["start"], e["end"], e["text"]) for e in preds}
        for span in gold_spans.intersection(pred_spans):
            buckets[get_bucket_name(span[2])]["Strict TP"] += 1
        for span in pred_spans - gold_spans:
            buckets[get_bucket_name(span[2])]["Strict FP"] += 1
        for span in gold_spans - pred_spans:
            buckets[get_bucket_name(span[2])]["Strict FN"] += 1

        # Relaxed (Overlap) Matching evaluation
        matched_preds = [False] * len(preds)
        matched_golds = [False] * len(golds)

        for p_idx, p in enumerate(preds):
            for g_idx, g in enumerate(golds):
                if not matched_golds[g_idx] and max(p["start"], g["start"]) < min(p["end"], g["end"]):
                    bucket = get_bucket_name(g["text"])
                    buckets[bucket]["Relaxed TP"] += 1
                    matched_preds[p_idx] = True
                    matched_golds[g_idx] = True
                    break 

        for p_idx, p in enumerate(preds):
            if not matched_preds[p_idx]:
                bucket = get_bucket_name(p["text"])
                buckets[bucket]["Relaxed FP"] += 1

        for g_idx, g in enumerate(golds):
            if not matched_golds[g_idx]:
                bucket = get_bucket_name(g["text"])
                buckets[bucket]["Relaxed FN"] += 1

    # F1-Score calculation and DataFrame conversion
    results = []
    for b_name, c in buckets.items():
        for m_type in ["Strict", "Relaxed"]:
            tp = c[f"{m_type} TP"]
            fp = c[f"{m_type} FP"]
            fn = c[f"{m_type} FN"]
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
            results.append({"Model": model_name, "Token Bucket": b_name, 
                            "Match Type": f"{m_type} Match", "F1-Score": round(f1, 4)})
    return pd.DataFrame(results)

# Execution block for multiple models
if __name__ == '__main__':
    gemini_df = analyze_fragmentation_final(
        gold_file='project/test_hard.json', 
        pred_file='project/gemini_predictions_hard.json', 
        token_csv='project/token_analysis.csv', 
        model_name='Gemini (Zero-Shot)'
    )
    
    biobert_df = analyze_fragmentation_final(
        gold_file='project/test_hard.json', 
        pred_file='project/biobert_predictions_hard.json', 
        token_csv='project/token_analysis.csv', 
        model_name='BioBERT'
    )

    bert_df = analyze_fragmentation_final(
        gold_file='project/test_hard.json', 
        pred_file='project/bert_predictions_hard.json', 
        token_csv='project/token_analysis.csv', 
        model_name='Standard BERT'
    )

    final_df = pd.concat([bert_df, biobert_df, gemini_df])

    print("=== FINAL DUAL FRAGMENTATION RESULTS ===")
    print(final_df.to_string(index=False))